# **Gemma 4 26B-A4B Memorilla 10% Subsampled Curriculum Guide**

## **Preface: Context & Architecture Overview**

This guide outlines the end-to-end process of adapting and training the **Memorilla** architecture (`snap-stanford/memorilla`) across its complete three-stage curriculum for Google DeepMind's **Gemma 4 26B-A4B** (`google/gemma-4-26B-A4B-it`) large language model on a single-device unified-memory workstation (AMD Strix Halo Radeon 8060S with a 64 GB VRAM allocation on Windows ROCm) \[1, 4\].

## **What is Memorilla?**

Memorilla is a framework that introduces a **latent semantic memory** interface for frozen LLMs \[1, 2\]. Instead of prepending long, raw text passages to an LLM's prompt—which spikes inference compute costs and KV-cache memory usage—Memorilla encodes external corpus documents once using a frozen document encoder (`Qwen/Qwen3-Embedding-4B`) into reusable, persistent embeddings. When a user submits a query, a set of $K = 16$ learned memory slots ($Q_0 \in \mathbb{R}^{K \times h_{\text{enc}}}$) first conditions on the query embedding via cross-attention, then refines its representations across all stored document embeddings using Perceiver-style cross-attention blocks. Finally, a learned linear projection maps these vectors into the frozen LLM's input embedding space as 16 soft "memory tokens," allowing the model to recall and synthesize deep factual context with up to $394\times$ fewer source document tokens \[1\].

## **Why This Curriculum and Subsampling?**

The official Memorilla training pipeline progresses through a three-stage curriculum (`stage1_enwiki`, `stage2_mixture`, and `stage3_multitask`), backed by roughly 310 GiB of text splits and precomputed `Qwen3-Embedding-4B` vectors hosted in `memorilla/Memorilla-Data` \[2, 3\]:

| Stage & Recipe | Target Datasets | Role in Curriculum | Official Schedule (4x H100) | 10% Gemma 4 Schedule (1x Strix Halo) |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** ([`stage1_gemma4_10pct.yaml`](./configs/stage1_gemma4_10pct.yaml)) | `enwiki` ($\sim 1.98\text{M}$ passages) | Pretrains memory slots to carry document content to the decoder via paraphrase restatement. | 1 epoch, LR $3\times 10^{-4}$, Batch $4 \times 8 \times 4 = 128$ ($\sim 6\text{ hrs}$) | **Projection Alignment:** Freeze Perceiver (`MEMORILLA_FREEZE_PERCEIVER=1`), train $2560 \to 2816$ projection on 10% split (1 epoch, LR $5\times 10^{-4}$, Batch $4 \times 32 \times 1 = 128$) |
| **Stage 2** ([`stage2_gemma4_10pct.yaml`](./configs/stage2_gemma4_10pct.yaml)) | `squad_v2`, `drop`, `coqa`, `pubmedqa`, `quail`, `pwc`, `cnn_dailymail`, `samsum`, `dialogsum`, `msmarco` ($1.34\text{M}$ rows) | Teaches the module to recall diverse task-specific information from the embedding store. | 1 epoch, LR $3\times 10^{-4}$, Batch $4 \times 8 \times 4 = 128$ ($\sim 4\text{ hrs}$) | **General QA Adaptation:** Unfreeze full `MemoryModule` on 10% mixture with 2,500-row floor (1 epoch, LR $1.5\times 10^{-4}$, Batch $4 \times 32 \times 1 = 128$) |
| **Stage 3** ([`stage3_gemma4_10pct.yaml`](./configs/stage3_gemma4_10pct.yaml)) | `pv4`, `pmv2`, `factkg`, `triviaqa`, `narrativeqa`, `pubmedqa`, `lamp4`, `lamp7` | Fine-tunes one shared module across all 8 downstream evaluation benchmarks, saving every epoch for per-benchmark validation selection. | 5 epochs, LR $1\times 10^{-4}$, Batch $2 \times 8 \times 4 = 64$ ($\sim 9\text{ hrs}$) | **Downstream Multi-Task:** Full module fine-tuning across 8 benchmarks with 2,500-row floor (5 epochs, LR $1\times 10^{-4}$, Batch $2 \times 32 \times 1 = 64$) |

For developers operating on single-device unified-memory workstations or performing rapid cross-model transfer, training on 100% of Stages 1 and 2 ($\sim 3.32\text{M}$ total examples) is computationally prohibitive. This guide uses a **10% subsampled curriculum with a 2,500-row minimum floor per task**: massive corpora (`enwiki`, `msmarco`, `triviaqa`, `factkg`, `squad_v2`) are compressed to 10%, while compact Stage 2/3 splits (`pubmedqa`, `pv4`, `pmv2`, `narrativeqa`, `lamp4`, `lamp7`) retain at least 2,500 training rows (or 100% if smaller) so personalization and domain QA tasks are never data-starved. All splits preserve Memorilla's native Parquet schema (`collection_id`, `question`, `answer`, `documents`) and link directly to the precomputed embedding store so `train.py` never recomputes document embeddings from scratch.

## **Why Gemma 4 26B-A4B?**

The official Memorilla checkpoints (`memorilla/Memorilla-Qwen3-8B`) were trained against a frozen `Qwen/Qwen3-8B` decoder ($h_{\text{dec}} = 4096$) paired with a frozen `Qwen/Qwen3-Embedding-4B` encoder ($h_{\text{enc}} = 2560$) \[2, 3\]. This workflow adapts the pretrained Perceiver memory module to target **Gemma 4 26B-A4B**—a Mixture-of-Experts (MoE) architecture with 25.2B total parameters that activates 3.8B parameters (8 of 128 routed experts plus 1 shared expert across 30 layers) per token during inference \[4\].

Adapting Memorilla to Gemma 4 on a 64 GB Windows ROCm system requires five concrete engineering adjustments:

1. **Projection Weight Surgery & Hook-Based Scale Alignment (`attach_gemma4_scale_hooks`):** Resizing the output projection layer in `memory.pt` from $(4096, 2560)$ to Gemma 4's language hidden dimension of $(2816, 2560)$ at standard fan-in variance $\frac{1}{\sqrt{2560}} \approx 0.01976$ in GPU memory for AdamW, while attaching a PyTorch forward hook that multiplies output activations by $\sqrt{2816} \approx 53.066$ \[4\]. On checkpoint save, a `state_dict` hook folds the $\sqrt{2816}$ multiplier directly into `memory.pt` on disk (and divides it back out when loading into a patched process), so both patched training runs and standalone unpatched inference/evaluation runs (`evaluate.py`, `MemoryVLLM`) produce identically scaled memory tokens.
2. **Split-Half Packed INT4 Expert Quantization & Zero-Loop `torch.bmm` MoE Dispatch (`PackedInt4ExpertParameter` & `_nf4_experts_forward`):** A full `bfloat16` load ($\sim 50.4\text{ GB}$ decoder + activations) exceeds a 64 GB VRAM allocation, while standard `bitsandbytes` (`load_in_4bit=True`) silently skips 91% of Gemma 4's weights because `Gemma4TextExperts` stores all 128 routed experts as fused 3D `nn.Parameter` tensors (`gate_up_proj` $[128, 1408, 2816]$ and `down_proj` $[128, 2816, 704]$) rather than `nn.Linear` layers. Furthermore, Hugging Face's default `@use_experts_implementation` (`grouped_mm_experts_forward`) fails on custom quantized modules, and per-expert Python loops with 64-bit `int64` codebook lookups create massive memory-bandwidth and driver-launch bottlenecks. Staging the model in CPU RAM (`device_map="cpu"`), stripping the unused $\sim 550\text{M}$ vision tower ($\sim 1.1\text{ GB}$ saved), packing the first and second halves of each 64-weight group into low/high `uint8` nibbles (`42.54 GB` $\to$ `11.30 GB`, $3.76\times$ compression), and executing all active experts per layer in **two vectorized `rocBLAS` `torch.bmm` calls** delivers a $>3\times$ training speedup while preserving full autograd back to `inputs_embeds`.
3. **Eval-Safe Layer Checkpointing & KV-Cache Stripping (`MEMORILLA_CKPT_EVERY_N_LAYERS=1`, `use_cache=False`):** Because `train.py` places the frozen decoder in `.eval()` mode (`self.llm.text_model.requires_grad_(False).eval()`), standard Hugging Face gradient checkpointing is bypassed (`self.training == False`) and `DynamicCache` remains active (`use_cache=True`), which otherwise causes backward recomputation to double-append tokens to `past_key_values` ($242 + 242 = 484$) and crash in SDPA. Explicitly disabling `use_cache`, clearing `past_key_values` during gradient-enabled passes, and wrapping each `Gemma4TextDecoderLayer` in `torch.utils.checkpoint` whenever `torch.is_grad_enabled()` is `True` keeps PyTorch peak VRAM at $\sim 23.8\text{--}25.8\text{ GB}$ ($\sim 43.5 / 63.8\text{ GB}$ total Dedicated GPU Memory in Windows Task Manager, with $0\text{ GB}$ Shared GPU Memory spillover).
4. **Floor-Protected Subsampling & Row-Aligned Precomputed Embeddings:** Subsampling 10% of training splits with a 2,500-row floor while slicing `train/questions.safetensors` to the exact sampled indices and linking precomputed document embedding stores via Windows NTFS junctions (`_winapi.CreateJunction` / `mklink /J`) eliminates on-the-fly `Qwen3-Embedding-4B` encoding overhead.
5. **Fault-Tolerant `last.ckpt` + Intra-Epoch Save/Resume (`CheckpointResumeManager` & `_Gemma4LightningSyncCallback`):** Synchronizing custom intra-epoch checkpoints (`resume_latest/trainer_state.pt` and root `memory.pt`) with PyTorch Lightning's native `last.ckpt` every 50 steps, every 15 minutes, and on `Ctrl+C` (`SIGINT`), re-applying resumed weights in `on_train_start` (so `--pretrained_memory` warm-start initialization never overwrites resumed progress), and gating DataLoader fast-forwarding strictly to `RandomSampler` (`train_dataloader`) enables seamless pause-and-resume across multi-day training runs.

---

## **Repository File Index & Reference Links**

### **Local Project Files**

* **Execution Workflow (Windows PowerShell):** [`example.bat.txt`](./example.bat.txt)
* **Dataset Subsampling & Embedding Alignment:** [`subsample_data.py`](./subsample_data.py)
* **Checkpoint Weight Surgery (`4096` $\to$ `2816`):** [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py)
* **Gemma 4 Runtime Monkey-Patch, INT4 MoE & Resume Manager:** [`apply_gemma4_patch.py`](./apply_gemma4_patch.py)
* **Training & Evaluation Entry Points:** [`train.py`](./train.py) | [`evaluate.py`](./evaluate.py) | [`test_pipeline.py`](./test_pipeline.py)
* **Curriculum YAML Recipes:**
  * Stage 1 (`enwiki` Alignment): [`configs/stage1_gemma4_10pct.yaml`](./configs/stage1_gemma4_10pct.yaml)
  * Stage 2 (`mixture` Adaptation): [`configs/stage2_gemma4_10pct.yaml`](./configs/stage2_gemma4_10pct.yaml)
  * Stage 3 (`multitask` Fine-Tuning): [`configs/stage3_gemma4_10pct.yaml`](./configs/stage3_gemma4_10pct.yaml)

### **External Official Resources**

> * **Research Paper:** [Memorilla arXiv Pre-print (arXiv:2610.21375)](https://arxiv.org/abs/2610.21375) \[1\]
> * **Source Code & Training Docs:** [Stanford SNAP's Memorilla GitHub](https://github.com/snap-stanford/memorilla) and [Training Guide (`docs/training.md`)](https://github.com/snap-stanford/memorilla/blob/main/docs/training.md) \[2\]
> * **Datasets & Pretrained Checkpoints:** [Memorilla on Hugging Face](https://huggingface.co/memorilla) (`memorilla/Memorilla-Data` and `memorilla/Memorilla-Qwen3-8B`) \[3\]
> * **Target Decoder Architecture:** [Gemma 4 26B-A4B-IT Hugging Face Hub](https://huggingface.co/google/gemma-4-26B-A4B-it) \[4\]

---

## **1. Dataset Subsampling Pipeline (Stages 1, 2, and 3)**

**Script:** [`subsample_data.py`](./subsample_data.py)

Unlike flat Hugging Face datasets, `snap-stanford/memorilla` organizes `memorilla/Memorilla-Data` by task configurations across all three stages, storing question-answer rows in Parquet files (`<repo>/data/<dataset>/train-00000-of-00001.parquet`) mapped via `collection_id` to precomputed `Qwen3-Embedding-4B` vectors \[2, 3\].

Running [`subsample_data.py`](./subsample_data.py) automates three steps:
1. **Per-Split Resolution (`ensure_available_splits`):** Requests `("train", "validation", "test")` and automatically falls back to per-split fetching for pretraining datasets (like `enwiki`) that omit `test` splits, while pulling precomputed `embeddings/*/<dataset>/*` shards from `memorilla/Memorilla-Data`.
2. **Floor-Protected 10% Subsampling (`min_train_rows=2500`):** Deterministically samples 10% of large training splits while enforcing a 2,500-row minimum floor (`min(len(df), max(2500, int(len(df) * fraction)))`) so compact Stage 2/3 benchmarks retain full supervision. Validation and test splits are copied verbatim for Stage 3 epoch selection.
3. **Row-Aligned Question Embeddings & Cross-Platform Store Linking (`sync_dataset_embeddings`):** Slices `train/questions.safetensors` to match the exact sampled training row indices and links document `embeddings/`, `corpus/`, and `collections/` using directory junctions (`_winapi.CreateJunction` / `mklink /J`) on Windows or symlinks on Linux/macOS. Completed datasets write a `.complete` marker for instant resume.

---

## **2. Weight Surgery & Standard-Variance Initialization**

**Script:** [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py)

In `memorilla/Memorilla-Qwen3-8B`, pretrained `MemoryModule` checkpoints are organized into stage-specific subdirectories (`stage1/`, `stage2/`, `stage3/`, `pv4/`, `pmv2/`), where each directory stores `memory.pt` and `config.json` \[2, 3\]. [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py) downloads the `stage2/` subfolder, preserves all $h_{\text{enc}} = 2560$ Perceiver cross-attention and query-slot parameters ($183.0\text{M}$ params), and replaces only the final decoder projection (`4096` $\to$ `2816`, $7.2\text{M}$ params).

### **Why We Decouple Weight Initialization from the $\sqrt{2816}$ Embedding Scale**

Gemma 4 scales its native token embeddings by $\sqrt{d_{\text{model}}} = \sqrt{2816} \approx 53.066$ inside `Gemma4TextScaledWordEmbedding` \[4\]. However, baking that $53.066\times$ multiplier directly into the in-memory parameter tensor $W_{\text{proj}}$ ($\text{std} = \frac{\sqrt{2816}}{\sqrt{2560}} \approx 1.049$) creates two optimization pathologies under AdamW across a 7-epoch curriculum:
* **Weight Decay Distortion:** Decoupled weight decay ($-\eta \lambda W$) penalizes weights proportional to their magnitude, shrinking $W_{\text{proj}}$ $53\times$ faster than standard layers during Stages 2 and 3.
* **Relative Step-Size Compression:** Because AdamW normalizes parameter updates to unit RMS before multiplying by the learning rate $\eta$, a standard learning rate ($\eta = 1.5\times 10^{-4}$) produces relative weight updates $53\times$ smaller when $\|W\| \sim 1.05$ than when $\|W\| \sim \frac{1}{\sqrt{2560}} \approx 0.01976$.

To avoid both issues, [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py) initializes $W_{\text{proj}}$ at standard fan-in variance ($\text{std} = \frac{1}{\sqrt{2560}} \approx 0.01976$), while `attach_gemma4_scale_hooks` in [`apply_gemma4_patch.py`](./apply_gemma4_patch.py) multiplies the projected output activation by $\sqrt{2816} \approx 53.066$ in the forward pass and folds the scale into `memory.pt` when saving to disk.

---

## **3. Runtime Monkey-Patch, Batched-GEMM INT4 MoE & Save/Resume Manager**

**Script:** [`apply_gemma4_patch.py`](./apply_gemma4_patch.py)

Running `python apply_gemma4_patch.py --memorilla_root .` injects `install_runtime_monkeypatch()` into the top of [`train.py`](./train.py) and [`evaluate.py`](./evaluate.py). It coordinates five runtime subsystems:

1. **Split-Half Packed INT4 Experts & Zero-Loop `torch.bmm` MoE Forward Pass (`PackedInt4ExpertParameter` & `_nf4_experts_forward`):**
   * **Split-Half `bfloat16` Unpacking:** Compresses all 60 fused 3D expert tensors across 30 `Gemma4TextExperts` layers (`42.54 GB` $\to$ `11.30 GB` in $\sim 90\text{s}$ on CPU) using symmetric `group_size=64` scales (which evenly divides `2816`, `1408`, and `704`). By packing the first 32 weights of each group in the low nibble and the second 32 in the high nibble, `__getitem__` unpacks directly via `bfloat16` bit-shifts and `torch.cat` without 64-bit `int64` index tables or `torch.stack` interleaving (eliminating $\sim 1.74\text{ TB}$ of `int64` memory traffic per micro-batch).
   * **Batched GEMM Active-Expert Dispatch:** Overrides `Gemma4TextExperts.forward` to bypass `transformers`'s `grouped_mm_experts_forward`. Routed tokens are grouped by unique active expert via `torch.unique` and scattered into a padded 3D tensor $[E_{\text{active}}, \text{max\_count}, H]$ so all active experts in a layer execute in **two `rocBLAS` `torch.bmm` calls** (`gate_up_proj` and `down_proj`) with zero Python loops.
2. **Eval-Safe Layer Checkpointing & KV-Cache Disabling (`load_gemma4_text_decoder`):**
   * Sets `config.use_cache = False` and `config.text_config.output_router_logits = False`, aliases `text_config` attributes (`hidden_size=2816`, `vocab_size=262144`) onto the top-level `Gemma4Config`, strips `vision_tower` and `multi_modal_projector` ($\sim 1.1\text{ GB}$ saved), and calls `model.enable_input_require_grads()`.
   * Wraps every $N$-th `Gemma4TextDecoderLayer.forward` (controlled by `MEMORILLA_CKPT_EVERY_N_LAYERS`, set to `1` for 64 GB allocations) with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` whenever `torch.is_grad_enabled()` is `True`, while stripping `past_key_values` so backward recomputation never double-appends tokens even though `text_model` is in `.eval()` mode. Also overrides `resize_token_embeddings` to default to `mean_resizing=False` (avoiding `hipSOLVER` Cholesky crashes on ROCm).
3. **Zero-Nesting Forward & `state_dict` Scaling Hooks (`attach_gemma4_scale_hooks`):**
   * Attaches a forward hook (`(out.float() * 53.066).to(out.dtype)`) and `state_dict` save/load hooks directly onto `MemoryModule.memory_projection` (`nn.Linear(2560, 2816)`). Saved `memory.pt` files on disk have $\sqrt{2816}$ folded into the weights for standalone compatibility, while `_unscale_on_load` divides it back out whenever `weight.float().std() > 0.1`.
4. **Environment-Driven Stage 1 Perceiver Freezing (`MEMORILLA_FREEZE_PERCEIVER=1`):**
   * Freezes all Perceiver cross-attention and query-slot parameters when `MEMORILLA_FREEZE_PERCEIVER=1` is set in Stage 1, leaving only the $7.2\text{M}$-parameter `memory_projection` layer trainable.
5. **PyTorch Lightning & Intra-Epoch Save/Resume (`CheckpointResumeManager` & `_Gemma4LightningSyncCallback`):**
   * Binds `optimizer.step` via `types.MethodType` (with `_with_counter = True` for PyTorch `LinearLR` / `SequentialLR` compatibility) to compute post-closure `GradNorm`, log `[Gemma4Metrics]` / `[Gemma4Heartbeat]`, and append telemetry to `<output_dir>/metrics_history.jsonl`.
   * Writes atomic (`*.tmp` $\to$ `os.replace`) checkpoints to `<output_dir>/resume_latest/trainer_state.pt`, `<output_dir>/memory.pt`, and PyTorch Lightning's native `<output_dir>/last.ckpt` every `MEMORILLA_SAVE_EVERY_STEPS` (default `50`), every `MEMORILLA_SAVE_EVERY_MINUTES` (default `15`), and immediately on `Ctrl+C` (`SIGINT`).
   * On resume, `_Gemma4LightningSyncCallback.on_train_start` re-applies the resumed `MemoryModule` state *after* `train.py`'s `_warm_start` so `--pretrained_memory` never overwrites mid-epoch progress, and `DataLoader.__iter__` fast-forwards only `RandomSampler` training loaders (never `val_dataloader`).

### **Measured VRAM Footprint on Strix Halo (64 GB BIOS Allocation)**

| Metric / Component | Measured Footprint | Notes |
| :--- | :--- | :--- |
| **Gemma 4 26B-A4B Decoder (Loaded on GPU)** | `16.4 GB` (PyTorch Allocated) | `11.30 GB` INT4 Routed Experts + `~5.1 GB` BF16 Attention, Shared Experts & Embeddings (`Qwen3-Embedding-4B` is not loaded into VRAM because `train.py` uses precomputed disk embeddings) |
| **PyTorch Peak Active VRAM (`MEMORILLA_CKPT_EVERY_N_LAYERS=1`)** | `23.8 – 25.8 GB` | Includes `MemoryModule` (`190M` params), AdamW states, and 1-layer transient `bfloat16` expert dequantization buffers (`1.52 GB`) |
| **Windows Task Manager Dedicated GPU Memory** | `43.5 / 63.8 GB` | Leaves **$>20\text{ GB}$ of free Dedicated VRAM headroom** |
| **Windows Task Manager Shared GPU Memory** | `0.4 / 47.6 GB` | Zero system-RAM spillover during training |

---

## **4. YAML Configuration Overrides (Stages 1, 2, and 3)**

All three YAML configurations use native `snap-stanford/memorilla` schema keys (`max_epochs`, `strategy: "auto"`, `num_workers: 0`, `devices: 1`, `attn_implementation: "sdpa"`) and scale `gradient_accumulation_steps: 32` to match the official effective batch sizes ($128$ for Stages 1 & 2; $64$ for Stage 3) \[2\]:

| Stage | Configuration File | `batch_size` $\times$ `grad_accum` | `max_epochs` | `learning_rate` | Perceiver State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** (`enwiki` 10%) | [`configs/stage1_gemma4_10pct.yaml`](./configs/stage1_gemma4_10pct.yaml) | $4 \times 32 = 128$ | `1` | `5.0e-4` | Frozen (`MEMORILLA_FREEZE_PERCEIVER=1`; only `7.2M` `W_proj` trained) |
| **Stage 2** (`mixture` 10%) | [`configs/stage2_gemma4_10pct.yaml`](./configs/stage2_gemma4_10pct.yaml) | $4 \times 32 = 128$ | `1` | `1.5e-4` | Unfrozen (`MEMORILLA_FREEZE_PERCEIVER=0`; full `190M` module trained) |
| **Stage 3** (`multitask` 10%) | [`configs/stage3_gemma4_10pct.yaml`](./configs/stage3_gemma4_10pct.yaml) | $2 \times 32 = 64$ | `5` | `1.0e-4` | Unfrozen (`MEMORILLA_FREEZE_PERCEIVER=0`; saves `epoch-00` .. `epoch-04`) |

> **Stage 1 Early-Convergence Tip:** Because Stage 1 freezes the Perceiver blocks and only aligns the single $2560 \to 2816$ linear projection matrix (`7.2M` parameters), it typically converges well before all 197,980 Wikipedia passages (`1,546` optimizer steps) are exhausted. Because `CheckpointResumeManager` continuously updates `.\checkpoints\stage1_gemma4\memory.pt` every 15 minutes (and on `Ctrl+C`), you can stop Stage 1 via `Ctrl+C` once `train_loss_step` plateaus (typically around 300–500 optimizer steps) and proceed directly to Stage 2.

---

## **5. Windows PowerShell Execution Workflow**

**Workflow Script:** [`example.bat.txt`](./example.bat.txt)

All environment variables, one-time data/weight preparation commands, 3-stage training commands, and validation/test evaluation loops are consolidated in [`example.bat.txt`](./example.bat.txt):

1. **Environment & Kernel Configuration (Section 1 of [`example.bat.txt`](./example.bat.txt)):**
   * Sets `$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"` (enabling fused Memory-Efficient SDPA kernels on Windows ROCm), `$env:MEMORILLA_CKPT_EVERY_N_LAYERS = "1"` (preventing Shared GPU Memory spillover), and rolling 15-minute / 50-step checkpoint intervals (`$env:MEMORILLA_SAVE_EVERY_MINUTES = "15"`, `$env:MEMORILLA_SAVE_EVERY_STEPS = "50"`).
2. **One-Time Preparation (Section 2 of [`example.bat.txt`](./example.bat.txt)):**
   * Runs [`apply_gemma4_patch.py`](./apply_gemma4_patch.py), subsamples the 3-stage datasets + precomputed embeddings into `.\data_10pct` via [`subsample_data.py`](./subsample_data.py), and builds the warm-start checkpoint in `.\checkpoints\gemma4_warmstart` via [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py).
3. **Stages 1, 2, and 3 Training (Sections 3–5 of [`example.bat.txt`](./example.bat.txt)):**
   * Runs [`train.py`](./train.py) sequentially across [`configs/stage1_gemma4_10pct.yaml`](./configs/stage1_gemma4_10pct.yaml), [`configs/stage2_gemma4_10pct.yaml`](./configs/stage2_gemma4_10pct.yaml), and [`configs/stage3_gemma4_10pct.yaml`](./configs/stage3_gemma4_10pct.yaml). Press `Ctrl+C` at any time to save an atomic checkpoint (`last.ckpt` + `resume_latest/trainer_state.pt` + `memory.pt`) and pause; re-running the same command resumes automatically from the last committed optimizer step.
4. **Validation Epoch Selection & Final Test Evaluation (Sections 6–7 of [`example.bat.txt`](./example.bat.txt)):**
   * Evaluates `epoch-00` through `epoch-04` on benchmark validation splits via [`evaluate.py`](./evaluate.py) and runs full test-split evaluation across all 8 downstream benchmarks.

---

## **References**

\[1\] Fallahpour, A., Idehpour, P., Kothapalli, V., Mounier, N., Wu, S., Pardis, S., & Leskovec, J. (2026). *Memorilla: Latent Semantic Memory for LLMs*. arXiv preprint [arXiv:2610.21375](https://arxiv.org/abs/2610.21375).

\[2\] Stanford SNAP Group (2026). *Memorilla Official Repository & Training Documentation (`docs/training.md`)*. GitHub: <https://github.com/snap-stanford/memorilla>.

\[3\] Memorilla Organization (2026). *Memorilla-Data & Memorilla-Qwen3-8B Checkpoints*. Hugging Face: <https://huggingface.co/memorilla>.

\[4\] Google DeepMind (2026). *Gemma 4 26B-A4B-IT Model Card & Configuration*. Hugging Face: <https://huggingface.co/google/gemma-4-26B-A4B-it>.