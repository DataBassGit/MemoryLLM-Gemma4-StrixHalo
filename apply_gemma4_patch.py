# Save as apply_gemma4_patch.py and run `python apply_gemma4_patch.py` inside the project root
import argparse
import gc
import glob
import importlib.util
import itertools
import json
import math
import os
import re
import signal
import sys
import time

# Guard against broken user-site torchaudio .pyd binaries on Windows before importing transformers
try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    _orig_find_spec = importlib.util.find_spec

    def _safe_find_spec(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _safe_find_spec

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

GEMMA4_HIDDEN_SIZE = 2816
GEMMA4_SCALE = math.sqrt(GEMMA4_HIDDEN_SIZE)  # ~53.066

# Information-theoretically optimal 4-bit NormalFloat (NF4) codebook for zero-mean normal weights
NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _alias_gemma4_text_config_attrs(config):
    """
    Copies top-level attributes (`hidden_size`, `vocab_size`, etc.) from `config.text_config`
    onto multimodal `Gemma4Config` objects and disables KV caching for teacher-forced training.
    """
    config.use_cache = False
    if hasattr(config, "text_config"):
        config.text_config.output_router_logits = False
        config.text_config.use_cache = False
        for attr in ("hidden_size", "vocab_size", "num_hidden_layers", "eos_token_id", "pad_token_id"):
            if not hasattr(config, attr) and hasattr(config.text_config, attr):
                setattr(config, attr, getattr(config.text_config, attr))
    return config


def attach_gemma4_scale_hooks(linear_layer: nn.Linear, scale: float = GEMMA4_SCALE) -> nn.Linear:
    """
    Keeps `linear_layer.weight` at standard 1/sqrt(h_enc) variance in memory for AdamW,
    multiplies the activation by `scale` (~53.066) in the forward pass, and folds `scale`
    into `memory.pt` when saving to disk so unpatched `evaluate.py` runs work out-of-the-box.
    """
    if getattr(linear_layer, "_gemma4_scaled", False):
        return linear_layer

    # 1. Multiply forward output by sqrt(2816) in FP32 during training/eval to avoid BF16 scalar rounding
    linear_layer.register_forward_hook(lambda mod, inp, out: (out.float() * scale).to(out.dtype))

    # 2. Fold sqrt(2816) into state_dict on save so saved checkpoints match Gemma 4's native norm
    def _scale_on_save(module, destination, prefix, local_metadata):
        for k in ("weight", "bias"):
            key = prefix + k
            if key in destination and destination[key] is not None:
                destination[key] = (destination[key].float() * scale).to(destination[key].dtype)

    linear_layer._register_state_dict_hook(_scale_on_save)

    # 3. Unscale by sqrt(2816) when loading an already-folded checkpoint back into a hooked model
    def _unscale_on_load(state_dict, prefix, *args):
        w_key = prefix + "weight"
        b_key = prefix + "bias"
        if w_key in state_dict and state_dict[w_key] is not None:
            # If weight std > 0.1, the checkpoint already has sqrt(2816) folded into it
            if state_dict[w_key].float().std().item() > 0.1:
                state_dict[w_key] = (state_dict[w_key].float() / scale).to(state_dict[w_key].dtype)
                if b_key in state_dict and state_dict[b_key] is not None:
                    state_dict[b_key] = (state_dict[b_key].float() / scale).to(state_dict[b_key].dtype)

    linear_layer._register_load_state_dict_pre_hook(_unscale_on_load)
    linear_layer._gemma4_scaled = True
    return linear_layer


class PackedInt4ExpertParameter(nn.Module):
    """
    Stores a 3D MoE expert tensor `[num_experts, out_dim, in_dim]` in split-half packed 4-bit
    `uint8` + `bfloat16` group scales (`group_size=64`). Unpacks directly in `bfloat16`
    without 64-bit `int64` index tables or `torch.stack` interleaving (15x less memory traffic).
    """

    def __init__(self, weight_bf16: torch.Tensor, group_size: int = 64):
        super().__init__()
        num_experts, out_dim, in_dim = weight_bf16.shape
        if in_dim % group_size != 0 or (group_size % 2 != 0):
            raise ValueError(f"in_dim ({in_dim}) must be divisible by even group_size ({group_size})")

        self.shape = weight_bf16.shape
        self.group_size = group_size
        half_group = group_size // 2

        w_grouped = weight_bf16.detach().float().view(num_experts, out_dim, in_dim // group_size, group_size)
        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
        q_uint4 = (torch.clamp(torch.round(w_grouped / scales), -8.0, 7.0).to(torch.int8) + 8).to(torch.uint8)

        # Pack first half of each group [0:32] in low nibble, second half [32:64] in high nibble
        low_nibble = q_uint4[..., :half_group]
        high_nibble = q_uint4[..., half_group:]
        packed = (high_nibble << 4) | (low_nibble & 0x0F)

        self.register_buffer("packed_weight", packed.contiguous(), persistent=False)
        self.register_buffer("scales", scales.squeeze(-1).to(torch.bfloat16).contiguous(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.packed_weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scales.dtype

    def __getitem__(self, expert_idx) -> torch.Tensor:
        if isinstance(expert_idx, torch.Tensor) and expert_idx.ndim == 0:
            expert_idx = int(expert_idx.item())
        packed_slice = self.packed_weight[expert_idx]
        scales_slice = self.scales[expert_idx].unsqueeze(-1)
        half_group = self.group_size // 2

        # Unpack directly into a single pre-allocated BF16 buffer to avoid torch.cat peak doubling
        out = torch.empty(
            (*packed_slice.shape[:-1], self.group_size),
            dtype=torch.bfloat16,
            device=packed_slice.device,
        )
        out[..., :half_group] = ((packed_slice & 0x0F).to(torch.bfloat16) - 8.0) * scales_slice
        out[..., half_group:] = ((packed_slice >> 4).to(torch.bfloat16) - 8.0) * scales_slice
        return out.flatten(-2)

    def transpose(self, dim0: int, dim1: int) -> torch.Tensor:
        return self[:].transpose(dim0, dim1)


def _nf4_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Chunked Batched GEMM (`torch.bmm`) MoE forward pass for quantized `Gemma4TextExperts`.
    Processes active experts in bounded chunks of 16 (`<= 126.8 MB` dequantized weight buffer
    instead of `2.03 GB` for all `E_active` experts at once), eliminating multi-GB variable-shape
    allocator fragmentation on Windows ROCm while preserving batched `rocBLAS` speed.
    """
    if top_k_index.is_floating_point():
        top_k_index, top_k_weights = top_k_weights, top_k_index

    orig_shape = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, orig_shape[-1])
    H = hidden_flat.shape[-1]
    top_k = top_k_index.shape[-1]
    flat_expert_ids = top_k_index.reshape(-1).long()
    flat_weights = top_k_weights.reshape(-1).to(hidden_flat.dtype)

    act_fn = getattr(self, "act_fn", lambda x: torch.nn.functional.gelu(x, approximate="tanh"))

    with torch.no_grad():
        active_experts, inv_idx, counts = torch.unique(
            flat_expert_ids, sorted=True, return_inverse=True, return_counts=True
        )
        perm = torch.argsort(inv_idx)
        sorted_token_idx = torch.div(perm, top_k, rounding_mode="floor")
        sorted_inv = inv_idx[perm]
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        pos_in_expert = torch.arange(perm.numel(), device=perm.device) - starts[sorted_inv]
        E_active = active_experts.numel()
        starts_cpu = starts.tolist()
        ends_cpu = ends.tolist()
        counts_cpu = counts.tolist()

    final_hidden_states = torch.zeros_like(hidden_flat)
    EXPERT_CHUNK = 16

    for start_e in range(0, E_active, EXPERT_CHUNK):
        end_e = min(start_e + EXPERT_CHUNK, E_active)
        chunk_experts = active_experts[start_e:end_e]
        C_e = end_e - start_e

        token_start = starts_cpu[start_e]
        token_end = ends_cpu[end_e - 1]
        chunk_max_count = max(counts_cpu[start_e:end_e])

        chunk_token_idx = sorted_token_idx[token_start:token_end]
        chunk_inv = sorted_inv[token_start:token_end] - start_e
        chunk_pos = pos_in_expert[token_start:token_end]
        chunk_w = flat_weights[perm[token_start:token_end]].unsqueeze(-1)

        padded_x = hidden_flat.new_zeros((C_e, chunk_max_count, H))
        padded_x[chunk_inv, chunk_pos] = hidden_flat[chunk_token_idx]

        gate_up_w = self.gate_up_proj[chunk_experts].to(hidden_flat.dtype)
        gate_up_out = torch.bmm(padded_x, gate_up_w.transpose(-1, -2))
        del gate_up_w, padded_x
        gate, up = gate_up_out.chunk(2, dim=-1)
        hidden_mid = act_fn(gate) * up
        del gate_up_out, gate, up

        down_w = self.down_proj[chunk_experts].to(hidden_flat.dtype)
        down_out = torch.bmm(hidden_mid, down_w.transpose(-1, -2))
        del down_w, hidden_mid

        valid_out = down_out[chunk_inv, chunk_pos] * chunk_w
        del down_out
        final_hidden_states.index_add_(0, chunk_token_idx, valid_out.to(final_hidden_states.dtype))
        del valid_out

    return final_hidden_states.reshape(orig_shape)


def quantize_gemma4_experts_to_int4(model: nn.Module, group_size: int = 64) -> None:
    """
    Replaces `gate_up_proj` and `down_proj` 3D parameters inside every `Gemma4TextExperts`
    module with split-half `PackedInt4ExpertParameter` modules and binds `_nf4_experts_forward`.
    """
    import types

    expert_modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Gemma4TextExperts"
    ]
    total_tensors = sum(
        1 for _, m in expert_modules for attr in ("gate_up_proj", "down_proj") if hasattr(m, attr)
    )
    converted = 0
    bf16_bytes_total = 0
    int4_bytes_total = 0
    t0 = time.monotonic()

    for _, module in expert_modules:
        for attr in ("gate_up_proj", "down_proj"):
            if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Parameter):
                param_bf16 = getattr(module, attr)
                bf16_bytes_total += param_bf16.numel() * param_bf16.element_size()
                delattr(module, attr)
                packed_mod = PackedInt4ExpertParameter(param_bf16, group_size=group_size)
                int4_bytes_total += (
                    packed_mod.packed_weight.numel()
                    + packed_mod.scales.numel() * packed_mod.scales.element_size()
                )
                setattr(module, attr, packed_mod)
                converted += 1
                if converted % 10 == 0 or converted == total_tensors:
                    pct = (converted / max(1, total_tensors)) * 100
                    print(
                        f"[Gemma4Patch] INT4 Quantization Progress: {converted}/{total_tensors} tensors ({pct:5.1f}%) | "
                        f"BF16 {bf16_bytes_total / (1024**3):.2f} GB -> INT4 {int4_bytes_total / (1024**3):.2f} GB "
                        f"({bf16_bytes_total / max(1, int4_bytes_total):.2f}x) | {time.monotonic() - t0:.1f}s"
                    )
        module.forward = types.MethodType(_nf4_experts_forward, module)


def load_gemma4_text_decoder(
    model_name_or_path: str = "google/gemma-4-26B-A4B-it",
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    int4_experts: bool = True,
    **kwargs,
):
    """
    Loads google/gemma-4-26B-A4B-it on CPU, strips vision modules, compresses the 128 MoE
    experts to 4-bit (~15.5 GB total decoder VRAM), checkpoints every Nth layer (default 3)
    to save 67% of backward recompute time on 64 GB setups, and moves the model to GPU.
    """
    import types

    target_device = kwargs.pop("device_map", "cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = kwargs.get("cache_dir")
    config = _alias_gemma4_text_config_attrs(AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir))

    if hasattr(config, "text_config"):
        config.text_config._experts_implementation = "eager"
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
            cache_dir=cache_dir,
        )
        inner = getattr(model, "model", model)
        for tower_attr in ["vision_tower", "audio_tower", "multi_modal_projector", "embed_vision", "embed_audio"]:
            if hasattr(inner, tower_attr):
                delattr(inner, tower_attr)

        if hasattr(inner, "language_model"):
            text_lm = inner.language_model
            if not hasattr(model, "get_input_embeddings") or model.get_input_embeddings() is None:
                model.get_input_embeddings = text_lm.get_input_embeddings
    else:
        config.output_router_logits = False
        config._experts_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
            cache_dir=cache_dir,
        )

    if int4_experts:
        quantize_gemma4_experts_to_int4(model, group_size=64)

    # Checkpoint every Nth decoder layer (default 3 -> 10 of 30 layers checkpointed) so 64 GB
    # setups avoid recomputing all 30 layers during backward while keeping VRAM around ~29 GB.
    # Default to checkpointing every layer ("1") so dequantized expert weights are never held across layers
    ckpt_every_n = int(os.environ.get("MEMORILLA_CKPT_EVERY_N_LAYERS", "1"))
    layer_idx = 0
    for module in model.modules():
        if module.__class__.__name__ == "Gemma4TextDecoderLayer":
            orig_layer_fwd = module.forward
            should_ckpt = (ckpt_every_n > 0) and (layer_idx % ckpt_every_n == 0)
            layer_idx += 1

            def _ckpt_layer_fwd(_self, *l_args, _orig=orig_layer_fwd, _do_ckpt=should_ckpt, **l_kwargs):
                if torch.is_grad_enabled():
                    l_kwargs["use_cache"] = False
                    l_kwargs["past_key_values"] = None
                    l_kwargs.pop("past_key_value", None)
                    if (
                        _do_ckpt
                        and len(l_args) > 0
                        and isinstance(l_args[0], torch.Tensor)
                        and l_args[0].requires_grad
                    ):
                        return torch.utils.checkpoint.checkpoint(_orig, *l_args, use_reentrant=False, **l_kwargs)
                return _orig(*l_args, **l_kwargs)

            module.forward = types.MethodType(_ckpt_layer_fwd, module)

    # Compute lm_head + softcapping + cross_entropy only on valid answer tokens (labels != -100)
    # in 128-token chunks (~134 MB fixed buffer) instead of [B, L, 262160] (6-14 GB variable buffer)
    from transformers.modeling_outputs import CausalLMOutputWithPast

    orig_model_fwd = model.forward

    def _chunked_loss_forward(self, *f_args, **f_kwargs):
        labels = f_kwargs.get("labels")
        backbone = getattr(self, "model", None)
        lm_head = getattr(self, "lm_head", None)
        if labels is None or backbone is None or lm_head is None:
            return orig_model_fwd(*f_args, **f_kwargs)

        b_kwargs = dict(f_kwargs)
        b_kwargs.pop("labels", None)
        b_kwargs["use_cache"] = False
        b_kwargs["return_dict"] = True

        hidden_states = backbone(*f_args, **b_kwargs)[0]
        shift_hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
        shift_labels = labels[..., 1:].reshape(-1)
        valid_mask = shift_labels != -100

        text_cfg = getattr(self.config, "text_config", self.config)
        softcap = getattr(text_cfg, "final_logit_softcapping", None)

        if valid_mask.any():
            valid_hidden = shift_hidden[valid_mask]
            valid_labels = shift_labels[valid_mask]
            total_loss = hidden_states.new_zeros((), dtype=torch.float32)

            def _chunk_ce(h_chunk: torch.Tensor, l_chunk: torch.Tensor) -> torch.Tensor:
                logits_c = lm_head(h_chunk)
                if softcap is not None:
                    logits_c = torch.tanh(logits_c / softcap) * softcap
                return torch.nn.functional.cross_entropy(logits_c.float(), l_chunk, reduction="sum")

            for i in range(0, valid_hidden.size(0), 128):
                h_c = valid_hidden[i : i + 128]
                l_c = valid_labels[i : i + 128]
                if torch.is_grad_enabled() and h_c.requires_grad:
                    chunk_loss = torch.utils.checkpoint.checkpoint(_chunk_ce, h_c, l_c, use_reentrant=False)
                else:
                    chunk_loss = _chunk_ce(h_c, l_c)
                total_loss = total_loss + chunk_loss

            loss = total_loss / valid_labels.numel()
        else:
            loss = (lm_head(shift_hidden[:1]).float().sum()) * 0.0

        if torch.is_grad_enabled():
            RESUME_MANAGER.record_micro_loss(loss)

        return CausalLMOutputWithPast(loss=loss, logits=None)

    model.forward = types.MethodType(_chunked_loss_forward, model)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "resize_token_embeddings"):
        orig_resize = model.resize_token_embeddings

        def _fast_resize(new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=False, *r_args, **r_kwargs):
            res = orig_resize(
                new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=mean_resizing,
                *r_args,
                **r_kwargs,
            )
            for p in model.parameters():
                p.requires_grad = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res

        model.resize_token_embeddings = _fast_resize

    gc.collect()
    if isinstance(target_device, str) and target_device != "auto":
        model.to(target_device)
    elif torch.cuda.is_available():
        model.to("cuda:0")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    secs = int(round(seconds))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def render_progress_bar(current: int, total: int, width: int = 12) -> str:
    if total <= 0:
        return f"[step {current}]"
    frac = min(1.0, max(0.0, current / total))
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {frac * 100:5.1f}%"


def resolve_run_metadata_from_cli(argv=None) -> dict:
    """
    Extracts `output_dir`, `epochs` (`max_epochs`), `batch_size`, and `gradient_accumulation_steps`
    from CLI flags or the YAML file passed to `--config`.
    """
    args = list(sys.argv if argv is None else argv)
    meta = {"output_dir": None, "epochs": None, "batch_size": None, "gradient_accumulation_steps": None}
    config_path = None

    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.split("#", 1)[0].strip()
                for k in ("output_dir", "max_epochs", "epochs", "batch_size", "gradient_accumulation_steps"):
                    if clean.startswith(f"{k}:"):
                        val = clean.split(":", 1)[1].strip().strip('"').strip("'")
                        dest_k = "epochs" if k in ("epochs", "max_epochs") else k
                        meta[dest_k] = val if dest_k == "output_dir" else int(float(val))

    for i, arg in enumerate(args):
        for k in ("output_dir", "max_epochs", "epochs", "batch_size", "gradient_accumulation_steps"):
            dest_k = "epochs" if k in ("epochs", "max_epochs") else k
            if arg == f"--{k}" and i + 1 < len(args):
                val = args[i + 1]
                meta[dest_k] = val if dest_k == "output_dir" else int(float(val))
            elif arg.startswith(f"--{k}="):
                val = arg.split("=", 1)[1]
                meta[dest_k] = val if dest_k == "output_dir" else int(float(val))
    return meta


def resolve_output_dir_from_cli(argv=None) -> str | None:
    """
    Extracts `output_dir` from CLI flags (`--output_dir`) or the YAML file passed to `--config`.
    """
    return resolve_run_metadata_from_cli(argv).get("output_dir")


class CheckpointResumeManager:
    """
    Provides fault-tolerant intra-epoch and inter-epoch save/resume plus live console telemetry:
    - Saves atomic (`*.tmp` -> `os.replace`) checkpoints to `<output_dir>/resume_latest/`
      every `save_every_steps` optimizer steps (default 50), every `save_every_minutes` (default 15m),
      at the end of each epoch, and immediately upon catching `SIGINT` (Ctrl+C) or `SIGTERM`.
    - Streams live console progress metrics (epoch/step progress bar, LR, grad norm, projector std,
      step throughput, elapsed time, epoch ETA, stage ETA, VRAM footprint, and next-save countdown)
      and appends structured records to `<output_dir>/metrics_history.jsonl`.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        save_every_steps: int | None = None,
        save_every_minutes: float | None = None,
        log_every_steps: int | None = None,
        log_every_micro_steps: int | None = None,
    ):
        self.reset(
            output_dir=output_dir,
            save_every_steps=save_every_steps,
            save_every_minutes=save_every_minutes,
            log_every_steps=log_every_steps,
            log_every_micro_steps=log_every_micro_steps,
        )

    def reset(
        self,
        output_dir: str | None = None,
        save_every_steps: int | None = None,
        save_every_minutes: float | None = None,
        log_every_steps: int | None = None,
        log_every_micro_steps: int | None = None,
    ) -> None:
        cli_meta = resolve_run_metadata_from_cli()
        self.output_dir = (
            output_dir
            or os.environ.get("MEMORILLA_OUTPUT_DIR")
            or os.environ.get("MEMORILLA_STAGE_DIR")
            or cli_meta.get("output_dir")
        )
        self.total_epochs = int(cli_meta.get("epochs") or os.environ.get("MEMORILLA_TOTAL_EPOCHS", "0") or 0)
        self.grad_accum = int(cli_meta.get("gradient_accumulation_steps") or 1)
        self.save_every_steps = int(
            save_every_steps
            if save_every_steps is not None
            else os.environ.get("MEMORILLA_SAVE_EVERY_STEPS", "50")
        )
        self.save_every_seconds = float(
            (save_every_minutes if save_every_minutes is not None else float(os.environ.get("MEMORILLA_SAVE_EVERY_MINUTES", "15")))
            * 60.0
        )
        self.log_every_steps = int(
            log_every_steps
            if log_every_steps is not None
            else os.environ.get("MEMORILLA_LOG_EVERY_STEPS", "1")
        )
        self.log_every_micro_steps = int(
            log_every_micro_steps
            if log_every_micro_steps is not None
            else os.environ.get("MEMORILLA_LOG_EVERY_MICRO_STEPS", "16")
        )
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None

        self.current_epoch: int = 0
        self._epoch_cursor: int = 0
        self.global_step: int = 0
        self.current_micro_step: int = 0
        self.committed_micro_step: int = 0
        self.total_micro_steps_in_epoch: int = 0

        self.resume_epoch: int = 0
        self.resume_micro_step: int = 0
        self.pending_state: dict | None = None
        self._model_restored_for_loop: bool = False
        self.run_start_time: float = time.monotonic()
        self.epoch_start_time: float = time.monotonic()
        self.last_step_time: float = time.monotonic()
        self.ema_step_duration_s: float | None = None
        self.last_grad_norm: float = 0.0
        self._step_loss_sum: float = 0.0
        self._step_loss_count: int = 0
        self.last_step_loss: float | None = None
        self.ema_step_loss: float | None = None
        self.last_save_time: float = time.monotonic()
        self._signals_installed: bool = False

        if self.output_dir:
            self._discover_existing_checkpoint()

    def _discover_existing_checkpoint(self) -> None:
        if not self.output_dir or not os.path.exists(self.output_dir):
            return

        # 1. Check highest completed epoch directory (`epoch-NN/memory.pt`)
        completed_epochs = []
        for ep_dir in sorted(glob.glob(os.path.join(self.output_dir, "epoch-*"))):
            m = re.search(r"epoch-(\d+)$", os.path.basename(ep_dir))
            mem_pt = os.path.join(ep_dir, "memory.pt")
            if m and os.path.exists(mem_pt):
                completed_epochs.append((int(m.group(1)), mem_pt))

        # 2. Check intra-epoch resume state (`resume_latest/trainer_state.pt`)
        state_pt = os.path.join(self.output_dir, "resume_latest", "trainer_state.pt")
        if os.path.exists(state_pt):
            try:
                state = torch.load(state_pt, map_location="cpu", weights_only=False)
                self.pending_state = state
                self.resume_epoch = int(state.get("epoch", 0))
                self.resume_micro_step = int(state.get("committed_micro_step", 0))
                self.current_epoch = self.resume_epoch
                self.global_step = int(state.get("global_step", 0))
                self.committed_micro_step = self.resume_micro_step
                if state.get("last_step_loss") is not None:
                    self.last_step_loss = float(state["last_step_loss"])
                if state.get("ema_step_loss") is not None:
                    self.ema_step_loss = float(state["ema_step_loss"])
                print(
                    f"[Gemma4Resume] Found resumable state in {state_pt}: "
                    f"epoch={self.resume_epoch}, global_step={self.global_step}, "
                    f"committed_micro_step={self.resume_micro_step}"
                )
                return
            except Exception as exc:
                print(f"[Gemma4Resume] Warning: Could not load {state_pt} ({exc}); checking epoch-* checkpoints.")

        if completed_epochs:
            last_ep, last_mem_pt = completed_epochs[-1]
            self.resume_epoch = last_ep + 1
            self.resume_micro_step = 0
            self.current_epoch = self.resume_epoch
            self.pending_state = {
                "epoch": self.resume_epoch,
                "global_step": 0,
                "committed_micro_step": 0,
                "memory_state_dict": torch.load(last_mem_pt, map_location="cpu", weights_only=True),
            }
            print(
                f"[Gemma4Resume] Found completed epoch-{last_ep:02d}/memory.pt; "
                f"will resume starting at epoch {self.resume_epoch}."
            )

    def register_model(self, model: nn.Module, output_dir: str | None = None) -> None:
        if output_dir and output_dir != self.output_dir:
            self.reset(
                output_dir=output_dir,
                save_every_steps=self.save_every_steps,
                save_every_minutes=self.save_every_seconds / 60.0,
            )
        self.model = model
        self._model_restored_for_loop = False
        if self.pending_state and "memory_state_dict" in self.pending_state:
            model.load_state_dict(dict(self.pending_state["memory_state_dict"]), strict=False)
            print(f"[Gemma4Resume] Restored MemoryModule weights for {self.output_dir}")
        self.install_signal_handlers()

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        if getattr(optimizer, "_gemma4_resume_hooked", False):
            return optimizer
        import types

        self.optimizer = optimizer
        if self.pending_state and self.pending_state.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(self.pending_state["optimizer_state_dict"])
                print("[Gemma4Resume] Restored AdamW optimizer state.")
            except Exception as exc:
                print(f"[Gemma4Resume] Warning: Could not restore optimizer state ({exc}).")

        orig_step = optimizer.step

        def _wrapped_step(opt_self, *args, **kwargs):
            opt_self._opt_called = True
            result = orig_step(*args, **kwargs)
            grad_norm = self._compute_grad_norm()
            if grad_norm > 0.0:
                self.last_grad_norm = grad_norm
            self.on_optimizer_step()
            return result

        _wrapped_step._with_counter = True
        optimizer.step = types.MethodType(_wrapped_step, optimizer)
        optimizer._gemma4_resume_hooked = True
        return optimizer

    def _compute_grad_norm(self) -> float:
        if self.model is None:
            return 0.0
        sq_sum = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g_norm = p.grad.detach().float().norm(2).item()
                sq_sum += g_norm * g_norm
        return math.sqrt(sq_sum)

    def _compute_proj_std(self) -> float:
        if self.model is None:
            return 0.0
        for m in self.model.modules():
            if isinstance(m, nn.Linear) and tuple(m.weight.shape) in [(2816, 2560), (2560, 2816), (2816, 2816)]:
                return m.weight.detach().float().std().item()
        for p in self.model.parameters():
            return p.detach().float().std().item()
        return 0.0

    def _get_current_lr(self) -> float:
        if self.optimizer is not None and self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    def record_micro_loss(self, loss: torch.Tensor) -> None:
        if not torch.is_grad_enabled():
            return
        val = float(loss.detach().float().item())
        if math.isfinite(val):
            self._step_loss_sum += val
            self._step_loss_count += 1

    def log_step_metrics(self, step_dt: float) -> None:
        now = time.monotonic()
        elapsed_s = now - self.run_start_time
        rate_steps_per_s = 1.0 / max(1e-6, self.ema_step_duration_s or step_dt)

        # Compute Epoch & Stage ETAs when DataLoader length is known
        total_m = self.total_micro_steps_in_epoch
        cur_m = self.committed_micro_step
        ep_bar = render_progress_bar(cur_m, total_m, width=12)
        ep_label = (
            f"Ep {self.current_epoch + 1:02d}/{self.total_epochs:02d}"
            if self.total_epochs > 0
            else f"Ep {self.current_epoch + 1:02d}"
        )

        if total_m > 0 and cur_m > 0:
            msteps_per_step = float(self.grad_accum) if self.grad_accum > 1 else max(1.0, cur_m / max(1, self.global_step))
            msteps_per_s = rate_steps_per_s * msteps_per_step
            rem_m_epoch = max(0, total_m - cur_m)
            eta_epoch_s = rem_m_epoch / max(1e-6, msteps_per_s)
            if self.total_epochs > self.current_epoch + 1:
                rem_epochs = self.total_epochs - (self.current_epoch + 1)
                eta_stage_s = eta_epoch_s + (rem_epochs * total_m) / max(1e-6, msteps_per_s)
            else:
                eta_stage_s = eta_epoch_s
            mstep_str = f"mstep {cur_m}/{total_m}"
        else:
            eta_epoch_s = float("nan")
            eta_stage_s = float("nan")
            mstep_str = f"mstep {cur_m}"

        lr = self._get_current_lr()
        proj_std = self._compute_proj_std()
        steps_to_save = (
            self.save_every_steps - (self.global_step % self.save_every_steps)
            if self.save_every_steps > 0
            else 0
        )

        if torch.cuda.is_available():
            vram_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            vram_resv_gb = torch.cuda.memory_reserved() / (1024**3)
            mem_str = f"VRAM: {vram_alloc_gb:.1f}/{vram_peak_gb:.1f}GB (Resv: {vram_resv_gb:.1f}GB)"
        else:
            vram_alloc_gb = 0.0
            vram_peak_gb = 0.0
            mem_str = "Dev: CPU"

        loss_str = (
            f"Loss: {self.last_step_loss:.4f} (EMA: {self.ema_step_loss:.4f}) | "
            if self.last_step_loss is not None and self.ema_step_loss is not None
            else ""
        )

        print(
            f"[Gemma4Metrics] {ep_label} {ep_bar} | Step {self.global_step} ({mstep_str}) | "
            f"{loss_str}LR: {lr:.2e} | GradNorm: {self.last_grad_norm:.4f} | ProjStd: {proj_std:.5f} | "
            f"{step_dt:.2f}s/step | Elapsed: {format_duration(elapsed_s)} | "
            f"EpETA: {format_duration(eta_epoch_s)} | StageETA: {format_duration(eta_stage_s)} | "
            f"{mem_str} | SaveIn: {steps_to_save}st"
        )

        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                metrics_path = os.path.join(self.output_dir, "metrics_history.jsonl")
                record = {
                    "timestamp": time.time(),
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "micro_step": cur_m,
                    "total_micro_steps": total_m,
                    "pct_epoch": round((cur_m / total_m) * 100.0, 2) if total_m > 0 else None,
                    "step_loss": round(self.last_step_loss, 6) if self.last_step_loss is not None else None,
                    "ema_loss": round(self.ema_step_loss, 6) if self.ema_step_loss is not None else None,
                    "lr": lr,
                    "grad_norm": round(self.last_grad_norm, 6),
                    "proj_weight_std": round(proj_std, 6),
                    "step_time_s": round(step_dt, 4),
                    "elapsed_s": round(elapsed_s, 2),
                    "eta_epoch_s": round(eta_epoch_s, 2) if math.isfinite(eta_epoch_s) else None,
                    "eta_stage_s": round(eta_stage_s, 2) if math.isfinite(eta_stage_s) else None,
                    "vram_alloc_gb": round(vram_alloc_gb, 3),
                    "vram_peak_gb": round(vram_peak_gb, 3),
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError:
                pass

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler
        if self.pending_state and self.pending_state.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(self.pending_state["scheduler_state_dict"])
                print("[Gemma4Resume] Restored LRScheduler state.")
            except Exception as exc:
                print(f"[Gemma4Resume] Warning: Could not restore scheduler state ({exc}).")
        return scheduler

    def _ensure_model_restored_before_loop(self) -> None:
        if not self._model_restored_for_loop and self.model is not None and self.pending_state is not None:
            if "memory_state_dict" in self.pending_state:
                self.model.load_state_dict(dict(self.pending_state["memory_state_dict"]), strict=False)
            if self.pending_state.get("torch_rng_state") is not None:
                try:
                    torch.set_rng_state(self.pending_state["torch_rng_state"])
                except Exception:
                    pass
            self._model_restored_for_loop = True

    def on_optimizer_step(self) -> None:
        now = time.monotonic()
        step_dt = max(1e-6, now - self.last_step_time)
        self.last_step_time = now
        self.ema_step_duration_s = (
            step_dt if self.ema_step_duration_s is None else (0.2 * step_dt + 0.8 * self.ema_step_duration_s)
        )
        self.global_step += 1
        self.committed_micro_step = self.current_micro_step

        if self._step_loss_count > 0:
            self.last_step_loss = self._step_loss_sum / self._step_loss_count
            self.ema_step_loss = (
                self.last_step_loss
                if self.ema_step_loss is None
                else (0.1 * self.last_step_loss + 0.9 * self.ema_step_loss)
            )
            self._step_loss_sum = 0.0
            self._step_loss_count = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.log_every_steps > 0 and (self.global_step % self.log_every_steps == 0):
            self.log_step_metrics(step_dt)

        step_trigger = self.save_every_steps > 0 and (self.global_step % self.save_every_steps == 0)
        time_trigger = self.save_every_seconds > 0 and ((now - self.last_save_time) >= self.save_every_seconds)
        if step_trigger or time_trigger:
            self.save_checkpoint(reason=f"step_{self.global_step}")

    def save_checkpoint(self, reason: str = "manual") -> str | None:
        if self.model is None or not self.output_dir:
            return None

        resume_dir = os.path.join(self.output_dir, "resume_latest")
        os.makedirs(resume_dir, exist_ok=True)

        mem_sd = self.model.state_dict()
        state_payload = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "committed_micro_step": self.committed_micro_step,
            "last_step_loss": self.last_step_loss,
            "ema_step_loss": self.ema_step_loss,
            "memory_state_dict": mem_sd,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "reason": reason,
            "timestamp": time.time(),
        }

        state_pt = os.path.join(resume_dir, "trainer_state.pt")
        mem_pt = os.path.join(resume_dir, "memory.pt")
        root_mem_pt = os.path.join(self.output_dir, "memory.pt")

        # Write to .tmp files and os.replace atomically so Ctrl+C / power loss never corrupts checkpoints
        tmp_state = state_pt + ".tmp"
        tmp_mem = mem_pt + ".tmp"
        tmp_root_mem = root_mem_pt + ".tmp"
        torch.save(state_payload, tmp_state)
        os.replace(tmp_state, state_pt)
        torch.save(mem_sd, tmp_mem)
        os.replace(tmp_mem, mem_pt)
        torch.save(mem_sd, tmp_root_mem)
        os.replace(tmp_root_mem, root_mem_pt)

        # Also persist config.json at both root and resume_latest if the model has a MemoryConfig
        if hasattr(self.model, "config"):
            try:
                import dataclasses

                cfg_dict = dataclasses.asdict(self.model.config)
                for cfg_dir in (self.output_dir, resume_dir):
                    cfg_path = os.path.join(cfg_dir, "config.json")
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg_dict, f, indent=2)
                        f.write("\n")
            except Exception:
                pass

        self.last_save_time = time.monotonic()
        print(
            f"[Gemma4Resume] Saved atomic checkpoint ({reason}) -> {resume_dir} "
            f"[epoch={self.current_epoch}, global_step={self.global_step}, "
            f"committed_micro_step={self.committed_micro_step}]"
        )
        return state_pt

    def iterate_dataloader(self, raw_iterable):
        """
        Wraps a training DataLoader iterator to skip completed epochs (`epoch < resume_epoch`)
        and fast-forward past `committed_micro_step` batches when resuming mid-epoch.
        """
        epoch_idx = self._epoch_cursor
        if epoch_idx < self.resume_epoch:
            print(
                f"[Gemma4Resume] Skipping already completed epoch {epoch_idx} "
                f"(resuming at epoch {self.resume_epoch})"
            )
            self._epoch_cursor += 1
            return

        self._ensure_model_restored_before_loop()
        self.current_epoch = epoch_idx
        self.epoch_start_time = time.monotonic()
        self.last_step_time = time.monotonic()
        start_micro = self.resume_micro_step if epoch_idx == self.resume_epoch else 0
        self.current_micro_step = start_micro
        self.committed_micro_step = start_micro

        source_obj = raw_iterable() if callable(raw_iterable) else raw_iterable
        try:
            self.total_micro_steps_in_epoch = len(source_obj)  # type: ignore[arg-type]
        except TypeError:
            self.total_micro_steps_in_epoch = 0

        raw_iter = iter(source_obj)
        if start_micro > 0:
            print(
                f"[Gemma4Resume] Fast-forwarding epoch {epoch_idx} DataLoader by "
                f"{start_micro}/{self.total_micro_steps_in_epoch or '?'} micro-batches..."
            )
            if hasattr(raw_iter, "_sampler_iter"):
                raw_iter._sampler_iter = itertools.islice(raw_iter._sampler_iter, start_micro, None)
            else:
                raw_iter = itertools.islice(raw_iter, start_micro, None)

        for batch in raw_iter:
            self.current_micro_step += 1
            if (
                self.log_every_micro_steps > 0
                and self.current_micro_step % self.log_every_micro_steps == 0
                and self.current_micro_step != self.committed_micro_step
            ):
                bar = render_progress_bar(self.current_micro_step, self.total_micro_steps_in_epoch, width=10)
                run_loss_str = (
                    f" | RunLoss: {self._step_loss_sum / self._step_loss_count:.4f}"
                    if self._step_loss_count > 0
                    else ""
                )
                print(
                    f"[Gemma4Heartbeat] Ep {self.current_epoch + 1:02d} {bar} | "
                    f"Accumulating micro-batch {self.current_micro_step}/{self.total_micro_steps_in_epoch or '?'} "
                    f"(last committed step={self.global_step}){run_loss_str} | "
                    f"Elapsed: {format_duration(time.monotonic() - self.run_start_time)}"
                )
            yield batch

        # Epoch completed naturally: advance epoch cursor and reset micro-step offset
        self._epoch_cursor += 1
        self.current_epoch = self._epoch_cursor
        self.current_micro_step = 0
        self.committed_micro_step = 0
        self.resume_micro_step = 0
        self.save_checkpoint(reason=f"epoch_{epoch_idx:02d}_complete")

    def install_signal_handlers(self) -> None:
        if self._signals_installed:
            return
        try:
            def _handle_signal(signum, frame):
                print(
                    f"\n[Gemma4Resume] Caught signal {signum}! Saving atomic checkpoint "
                    f"at last committed optimizer step before exiting..."
                )
                self.save_checkpoint(reason=f"signal_{signum}")
                raise KeyboardInterrupt("Training paused cleanly via signal; checkpoint saved.")

            signal.signal(signal.SIGINT, _handle_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handle_signal)
            self._signals_installed = True
        except ValueError:
            # signal.signal only works in main thread
            pass


RESUME_MANAGER = CheckpointResumeManager()


def configure_gemma4_memory_module(
    memory_module: nn.Module,
    freeze_perceiver: bool = False,
    output_dir: str | None = None,
) -> nn.Module:
    """
    Attaches Gemma 4 forward/state_dict scaling hooks to the output projection layer,
    optionally freezes Perceiver blocks for Stage 1 projection alignment, and registers
    the module with RESUME_MANAGER for fault-tolerant save/resume.
    """
    proj_layer = None
    for name, module in memory_module.named_modules():
        if isinstance(module, nn.Linear) and tuple(module.weight.shape) in [(2816, 2560), (2560, 2816)]:
            proj_layer = module
            attach_gemma4_scale_hooks(module, scale=GEMMA4_SCALE)
            print(f"[Gemma4Patch] Attached sqrt(2816) scale hooks to MemoryModule.{name}")
            break

    if freeze_perceiver:
        for param in memory_module.parameters():
            param.requires_grad = False
        if proj_layer is not None:
            for param in proj_layer.parameters():
                param.requires_grad = True
        print("[Gemma4Patch] Stage 1 Mode: Perceiver frozen; only 2560 -> 2816 projector is trainable.")

    if output_dir or resolve_output_dir_from_cli() or os.environ.get("MEMORILLA_OUTPUT_DIR") or os.environ.get("MEMORILLA_STAGE_DIR"):
        RESUME_MANAGER.register_model(memory_module, output_dir=output_dir)
    return memory_module


def install_runtime_monkeypatch() -> None:
    """
    Monkey-patches AutoConfig.from_pretrained, AutoModelForCausalLM.from_pretrained (when called with gemma-4),
    MemoryModule initialization, AdamW optimizer, LRScheduler, DataLoader iteration, and Memorilla recipe/embedding
    helpers so both train.py and evaluate.py support Gemma 4 + automatic save/resume out of the box.
    """
    # Hard-cap PyTorch HIP allocator at 65% of 63.8 GB (~41.5 GB) so it never spills into
    # Windows WDDM Shared GPU Memory (System RAM) and C++ garbage_collection_threshold is active.
    if torch.cuda.is_available():
        vram_frac = float(os.environ.get("MEMORILLA_VRAM_FRACTION", "0.65"))
        try:
            torch.cuda.set_per_process_memory_fraction(vram_frac)
        except Exception:
            pass

    RESUME_MANAGER.reset()

    # 1. Alias text_config.hidden_size onto top-level Gemma4Config for AutoConfig.from_pretrained
    if not getattr(AutoConfig, "_gemma4_patched", False):
        orig_cfg_from_pretrained = AutoConfig.from_pretrained

        @classmethod
        def _patched_cfg_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            cfg = orig_cfg_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _alias_gemma4_text_config_attrs(cfg)

        AutoConfig.from_pretrained = _patched_cfg_from_pretrained
        AutoConfig._gemma4_patched = True

    # 2. Hook AutoModelForCausalLM.from_pretrained for Gemma 4 26B-A4B NF4 loading
    if not getattr(AutoModelForCausalLM, "_gemma4_patched", False):
        orig_from_pretrained = AutoModelForCausalLM.from_pretrained

        @classmethod
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            if "gemma-4" in str(pretrained_model_name_or_path).lower():
                return load_gemma4_text_decoder(pretrained_model_name_or_path, *args, **kwargs)
            return orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

        AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
        AutoModelForCausalLM._gemma4_patched = True

    # 3. Allow --pretrained_memory as a CLI alias for --init_memory in train.py (_ArgumentGroup & ArgumentParser)
    if not getattr(argparse._ActionsContainer, "_gemma4_patched", False):
        orig_add_arg = argparse._ActionsContainer.add_argument

        def _patched_add_arg(self, *args, **kwargs):
            if args == ("--init_memory",):
                args = ("--init_memory", "--pretrained_memory")
            return orig_add_arg(self, *args, **kwargs)

        argparse._ActionsContainer.add_argument = _patched_add_arg
        argparse._ActionsContainer._gemma4_patched = True

    # 4. Hook AdamW to auto-register with RESUME_MANAGER during train.py
    if not getattr(torch.optim.AdamW, "_gemma4_patched", False):
        orig_adamw_init = torch.optim.AdamW.__init__

        def _patched_adamw_init(self, *args, **kwargs):
            orig_adamw_init(self, *args, **kwargs)
            if RESUME_MANAGER.model is not None and RESUME_MANAGER.output_dir:
                RESUME_MANAGER.register_optimizer(self)

        torch.optim.AdamW.__init__ = _patched_adamw_init
        torch.optim.AdamW._gemma4_patched = True

    # 5. Hook LRScheduler to auto-register with RESUME_MANAGER during train.py
    if hasattr(torch.optim.lr_scheduler, "LRScheduler") and not getattr(
        torch.optim.lr_scheduler.LRScheduler, "_gemma4_patched", False
    ):
        orig_sched_init = torch.optim.lr_scheduler.LRScheduler.__init__

        def _patched_sched_init(self, *args, **kwargs):
            orig_sched_init(self, *args, **kwargs)
            if RESUME_MANAGER.model is not None and RESUME_MANAGER.output_dir:
                RESUME_MANAGER.register_scheduler(self)

        torch.optim.lr_scheduler.LRScheduler.__init__ = _patched_sched_init
        torch.optim.lr_scheduler.LRScheduler._gemma4_patched = True

    # 6. Hook DataLoader (disable pin_memory hipHostMalloc overhead on Windows & fast-forward batch_sampler)
    if not getattr(torch.utils.data.DataLoader, "_gemma4_patched", False):
        orig_dl_init = torch.utils.data.DataLoader.__init__
        orig_dl_iter = torch.utils.data.DataLoader.__iter__

        def _patched_dl_init(self, *args, **kwargs):
            if os.name == "nt":
                kwargs["pin_memory"] = False
            orig_dl_init(self, *args, **kwargs)

        # 6. Hook DataLoader (disable pin_memory hipHostMalloc overhead on Windows)
    if not getattr(torch.utils.data.DataLoader, "_gemma4_patched", False):
        orig_dl_init = torch.utils.data.DataLoader.__init__
        orig_dl_iter = torch.utils.data.DataLoader.__iter__

        def _patched_dl_init(self, *args, **kwargs):
            if os.name == "nt":
                kwargs["pin_memory"] = False
            orig_dl_init(self, *args, **kwargs)

        def _patched_dl_iter(self):
            is_train_sampler = isinstance(getattr(self, "sampler", None), torch.utils.data.RandomSampler)
            if (
                RESUME_MANAGER.model is not None
                and is_train_sampler
                and RESUME_MANAGER.output_dir
            ):
                return RESUME_MANAGER.iterate_dataloader(lambda: orig_dl_iter(self))
            return orig_dl_iter(self)

        torch.utils.data.DataLoader.__init__ = _patched_dl_init
        torch.utils.data.DataLoader.__iter__ = _patched_dl_iter
        torch.utils.data.DataLoader._gemma4_patched = True

    # 7. Hook PyTorch Lightning Trainer so Ctrl+C and step saves also write native `last.ckpt`
    try:
        import lightning.pytorch as pl

        if not getattr(pl.Trainer, "_gemma4_patched", False):
            orig_trainer_init = pl.Trainer.__init__

            class _Gemma4LightningSyncCallback(pl.Callback):
                def on_train_start(self, trainer, pl_module):
                    # Re-apply resumed weights AFTER train.py's _warm_start so --pretrained_memory never overwrites them
                    if RESUME_MANAGER.pending_state and "memory_state_dict" in RESUME_MANAGER.pending_state:
                        pl_module.memory.load_state_dict(
                            dict(RESUME_MANAGER.pending_state["memory_state_dict"]), strict=False
                        )
                        print(
                            f"[Gemma4Resume] Confirmed resumed weights active at "
                            f"global_step={RESUME_MANAGER.global_step}, mstep={RESUME_MANAGER.committed_micro_step}"
                        )

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    if (
                        RESUME_MANAGER.save_every_steps > 0
                        and RESUME_MANAGER.global_step > 0
                        and RESUME_MANAGER.current_micro_step == RESUME_MANAGER.committed_micro_step
                        and (RESUME_MANAGER.global_step % RESUME_MANAGER.save_every_steps == 0)
                        and RESUME_MANAGER.output_dir
                    ):
                        ckpt_path = os.path.join(RESUME_MANAGER.output_dir, "last.ckpt")
                        tmp_ckpt = ckpt_path + ".tmp"
                        trainer.save_checkpoint(tmp_ckpt)
                        os.replace(tmp_ckpt, ckpt_path)

                def on_exception(self, trainer, pl_module, exception):
                    if RESUME_MANAGER.output_dir and RESUME_MANAGER.global_step > 0:
                        ckpt_path = os.path.join(RESUME_MANAGER.output_dir, "last.ckpt")
                        tmp_ckpt = ckpt_path + ".tmp"
                        try:
                            trainer.save_checkpoint(tmp_ckpt)
                            os.replace(tmp_ckpt, ckpt_path)
                            print(f"[Gemma4Resume] Saved Lightning last.ckpt on interrupt -> {ckpt_path}")
                        except Exception:
                            pass

            def _patched_trainer_init(self, *args, **kwargs):
                cbs = list(kwargs.get("callbacks") or [])
                cbs.append(_Gemma4LightningSyncCallback())
                kwargs["callbacks"] = cbs
                orig_trainer_init(self, *args, **kwargs)

            pl.Trainer.__init__ = _patched_trainer_init
            pl.Trainer._gemma4_patched = True
    except ImportError:
        pass

    try:
        import dataclasses
        import memorilla
        import memorilla.embeddings
        import memorilla.utils

        # Normalize YAML recipes (map epochs -> max_epochs, strip bf16/gradient_checkpointing)
        if hasattr(memorilla.utils, "read_recipe") and not getattr(memorilla.utils.read_recipe, "_gemma4_patched", False):
            orig_read_recipe = memorilla.utils.read_recipe

            def _patched_read_recipe(path):
                recipe = dict(orig_read_recipe(path) or {})
                if "epochs" in recipe and "max_epochs" not in recipe:
                    recipe["max_epochs"] = recipe.pop("epochs")
                else:
                    recipe.pop("epochs", None)
                recipe.pop("bf16", None)
                recipe.pop("gradient_checkpointing", None)
                return recipe

            _patched_read_recipe._gemma4_patched = True
            memorilla.utils.read_recipe = _patched_read_recipe

        # Auto-sync precomputed embeddings in subsampled local repos without invoking vllm
        if hasattr(memorilla.embeddings, "ensure_embeddings") and not getattr(
            memorilla.embeddings.ensure_embeddings, "_gemma4_patched", False
        ):
            orig_ensure_emb = memorilla.embeddings.ensure_embeddings

            def _patched_ensure_emb(config, splits, repo=memorilla.embeddings.DATA_REPO, data_dir=memorilla.embeddings.DATA_DIR, encoder=memorilla.embeddings.DEFAULT_ENCODER, cache_dir=None):
                root = memorilla.embeddings.data_root(repo, data_dir)
                splits_list = list(splits)
                missing = [
                    s for s in splits_list
                    if memorilla.embeddings.missing_files(memorilla.embeddings.embedding_dir(config, s, root, encoder))
                ]
                if missing and memorilla.embeddings.local_repo(repo) is not None:
                    try:
                        import subsample_data

                        cache_root = subsample_data.ensure_available_splits(config, ("train", "validation", "test"))
                        src_data_dir = os.path.join(cache_root, "data", config)
                        indices = subsample_data.compute_deterministic_train_indices(src_data_dir, 0.10, 2500, 23)
                        subsample_data.sync_dataset_embeddings(config, cache_root, str(root), indices)
                    except Exception as exc:
                        print(f"[Gemma4Patch] Warning: Could not auto-sync embeddings for {config}: {exc}")
                return orig_ensure_emb(config, splits_list, repo=repo, data_dir=data_dir, encoder=encoder, cache_dir=cache_dir)

            _patched_ensure_emb._gemma4_patched = True
            memorilla.embeddings.ensure_embeddings = _patched_ensure_emb

        # [GEMMA4_NO_MMAP_EMBEDDING_STORE]
        # Replace safetensors mmap handles in EmbeddingStore with direct file seek+readinto
        # to prevent Windows 0xC0000005 STATUS_ACCESS_VIOLATION across 250GB+ Stage 3 shards.
        if hasattr(memorilla.embeddings, "EmbeddingStore") and not getattr(
            memorilla.embeddings.EmbeddingStore, "_gemma4_no_mmap", False
        ):
            import json as _st_json
            import struct as _st_struct

            _ST_DTYPE_MAP = {
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "F32": torch.float32,
                "I64": torch.int64,
                "I32": torch.int32,
                "I16": torch.int16,
                "I8": torch.int8,
                "U8": torch.uint8,
                "BOOL": torch.bool,
            }

            def _st_read_header(filepath):
                with open(filepath, "rb") as fp:
                    header_len = _st_struct.unpack("<Q", fp.read(8))[0]
                    header = _st_json.loads(fp.read(header_len).decode("utf-8"))
                base_offset = 8 + header_len
                meta = {}
                for k, v in header.items():
                    if k == "__metadata__":
                        continue
                    dtype = _ST_DTYPE_MAP[v["dtype"]]
                    shape = list(v["shape"])
                    s_off, e_off = v["data_offsets"]
                    nbytes = e_off - s_off
                    nrows = shape[0] if shape else 1
                    row_bytes = nbytes // max(1, nrows)
                    meta[k] = (base_offset + s_off, dtype, shape, row_bytes)
                return meta

            def _st_read_slice(filepath, meta_tuple, start_row, num_rows):
                byte_offset, dtype, shape, row_bytes = meta_tuple
                out_shape = [num_rows] + shape[1:]
                out = torch.empty(out_shape, dtype=dtype, device="cpu")
                nbytes = num_rows * row_bytes
                if nbytes > 0:
                    buf = out.view(torch.uint8).numpy()
                    view = memoryview(buf).cast("B")
                    with open(filepath, "rb") as fp:
                        fp.seek(byte_offset + start_row * row_bytes)
                        read_total = 0
                        while read_total < nbytes:
                            n = fp.readinto(view[read_total:])
                            if not n:
                                raise EOFError(f"Unexpected EOF reading {filepath} at row {start_row}")
                            read_total += n
                return out

            def _es_init(self, config, split, root=None, encoder=memorilla.embeddings.DEFAULT_ENCODER):
                self.config = config
                self.split = split
                self.directory = memorilla.embeddings.embedding_dir(
                    config, split, root if root is not None else memorilla.embeddings.data_root(), encoder
                )
                missing = memorilla.embeddings.missing_files(self.directory)
                if missing:
                    raise FileNotFoundError(f"Embeddings of {config}/{split} in {self.directory} lack {missing}.")
                idx_path = self.directory / memorilla.embeddings.INDEX_FILE
                idx_meta = _st_read_header(idx_path)
                self.collection_ids = _st_read_slice(idx_path, idx_meta["collection_ids"], 0, idx_meta["collection_ids"][2][0])
                self._shard = _st_read_slice(idx_path, idx_meta["shard"], 0, idx_meta["shard"][2][0])
                self._start = _st_read_slice(idx_path, idx_meta["start"], 0, idx_meta["start"][2][0])
                self._length = _st_read_slice(idx_path, idx_meta["length"], 0, idx_meta["length"][2][0])
                self._handles = {}

            def _es_open_meta(self, filename):
                if filename not in self._handles:
                    self._handles[filename] = _st_read_header(self.directory / filename)
                return self._handles[filename]

            @property
            def _es_num_questions(self):
                meta = self._open(memorilla.embeddings.QUESTIONS_FILE)[memorilla.embeddings.TENSOR_NAME]
                return int(meta[2][0])

            def _es_documents(self, collection_id):
                position = self._position(collection_id)
                if position is None:
                    raise KeyError(f"Collection {collection_id} not found in {self.config}/{self.split}.")
                shard = int(self._shard[position])
                start = int(self._start[position])
                length = int(self._length[position])
                fname = memorilla.embeddings.SHARD_TEMPLATE.format(shard)
                meta = self._open(fname)[memorilla.embeddings.TENSOR_NAME]
                return _st_read_slice(self.directory / fname, meta, start, length)

            def _es_question(self, row):
                fname = memorilla.embeddings.QUESTIONS_FILE
                meta = self._open(fname)[memorilla.embeddings.TENSOR_NAME]
                return _st_read_slice(self.directory / fname, meta, int(row), 1)[0]

            def _es_questions(self):
                fname = memorilla.embeddings.QUESTIONS_FILE
                meta = self._open(fname)[memorilla.embeddings.TENSOR_NAME]
                return _st_read_slice(self.directory / fname, meta, 0, int(meta[2][0]))

            memorilla.embeddings.EmbeddingStore.__init__ = _es_init
            memorilla.embeddings.EmbeddingStore._open = _es_open_meta
            memorilla.embeddings.EmbeddingStore.num_questions = _es_num_questions
            memorilla.embeddings.EmbeddingStore.documents = _es_documents
            memorilla.embeddings.EmbeddingStore.question = _es_question
            memorilla.embeddings.EmbeddingStore.questions = _es_questions
            memorilla.embeddings.EmbeddingStore._gemma4_no_mmap = True

        # Disable pin_memory on unified-memory Strix Halo APU to avoid HIP host page-locking pressure
        if hasattr(torch.utils.data, "DataLoader") and not getattr(torch.utils.data.DataLoader, "_gemma4_no_pin", False):
            _orig_dl_init = torch.utils.data.DataLoader.__init__

            def _patched_dl_init(self, *args, **kwargs):
                kwargs["pin_memory"] = False
                _orig_dl_init(self, *args, **kwargs)

            torch.utils.data.DataLoader.__init__ = _patched_dl_init
            torch.utils.data.DataLoader._gemma4_no_pin = True

        if hasattr(memorilla, "MemoryConfig") and not getattr(memorilla.MemoryConfig, "_gemma4_patched", False):
            orig_mc_init = memorilla.MemoryConfig.__init__
            valid_fields = {f.name for f in dataclasses.fields(memorilla.MemoryConfig)}

            def _patched_mc_init(self, *args, **kwargs):
                filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
                orig_mc_init(self, *args, **filtered)

            memorilla.MemoryConfig.__init__ = _patched_mc_init
            memorilla.MemoryConfig._gemma4_patched = True

        if hasattr(memorilla, "MemoryModule") and not getattr(memorilla.MemoryModule, "_gemma4_patched", False):
            orig_init = memorilla.MemoryModule.__init__

            def _patched_mm_init(self, *args, **kwargs):
                freeze_flag = kwargs.pop(
                    "freeze_perceiver",
                    os.environ.get("MEMORILLA_FREEZE_PERCEIVER", "0") == "1",
                )
                orig_init(self, *args, **kwargs)
                configure_gemma4_memory_module(self, freeze_perceiver=freeze_flag)

            memorilla.MemoryModule.__init__ = _patched_mm_init
            memorilla.MemoryModule._gemma4_patched = True
    except ImportError:
        pass


def patch_scripts(memorilla_root: str = ".") -> None:
    marker = "# [GEMMA4_MEMORILLA_PATCH]"
    injection = f"{marker}\nimport apply_gemma4_patch\napply_gemma4_patch.install_runtime_monkeypatch()\n"
    for script_name in ("train.py", "evaluate.py"):
        script = os.path.join(memorilla_root, script_name)
        if not os.path.exists(script) and os.environ.get("MEMORILLA_FETCH_SCRIPTS", "1") == "1":
            try:
                import urllib.request

                url = f"https://raw.githubusercontent.com/snap-stanford/memorilla/main/{script_name}"
                with urllib.request.urlopen(url, timeout=15) as resp:
                    if resp.status == 200:
                        with open(script, "w", encoding="utf-8") as f:
                            f.write(resp.read().decode("utf-8"))
                        print(f"[Gemma4Patch] Downloaded entry point {script} from snap-stanford/memorilla")
            except Exception as exc:
                print(f"[Gemma4Patch] Note: Could not auto-download {script} ({exc})")

        if os.path.exists(script):
            with open(script, "r", encoding="utf-8") as f:
                src = f.read()
            if marker not in src:
                with open(script, "w", encoding="utf-8") as f:
                    f.write(injection + src)
                print(f"[Gemma4Patch] Activated runtime monkey-patch in {script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject Gemma 4 runtime monkeypatch into train.py and evaluate.py")
    parser.add_argument("--memorilla_root", type=str, default=".")
    cli_args, _ = parser.parse_known_args()
    patch_scripts(cli_args.memorilla_root)
