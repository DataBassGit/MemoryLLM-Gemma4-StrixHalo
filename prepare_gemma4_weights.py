# Save as prepare_gemma4_weights.py
import os
import json
import math
import time
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def perform_gemma4_surgery(source_stage: str = "stage2", force: bool = False) -> None:
    """
    Downloads the pretrained Stage 2 MemoryModule from memorilla/Memorilla-Qwen3-8B
    and resizes its output projection from Qwen3-8B (4096) to Gemma 4 (2816) at
    standard 1/sqrt(h_enc) variance. The sqrt(2816) (~53.066) embedding scale is
    applied via PyTorch forward/state_dict hooks by `attach_gemma4_scale_hooks`
    in `apply_gemma4_patch.py`. Supports idempotent resume if already completed.
    """
    t0 = time.monotonic()
    output_dir = "checkpoints/gemma4_warmstart"
    out_pt = os.path.join(output_dir, "memory.pt")
    complete_marker = os.path.join(output_dir, ".complete")
    force_run = force or (os.environ.get("MEMORILLA_FORCE_SURGERY", "0") == "1")

    if not force_run and os.path.exists(out_pt) and os.path.exists(complete_marker):
        size_mb = os.path.getsize(out_pt) / (1024 * 1024)
        print(
            f"=== [Resume] Warm-start checkpoint already exists at {out_pt} "
            f"({size_mb:.2f} MB); skipping weight surgery ==="
        )
        return

    print(f"=== Commencing Checkpoint Transformation from {source_stage}/ ===")
    ckpt_root = snapshot_download(
        repo_id="memorilla/Memorilla-Qwen3-8B",
        allow_patterns=[f"{source_stage}/*"],
    )
    stage_dir = os.path.join(ckpt_root, source_stage)

    # Locate memory.pt (official Memorilla format) or .safetensors fallback
    pt_path = os.path.join(stage_dir, "memory.pt")
    safetensor_files = [f for f in os.listdir(stage_dir) if f.endswith(".safetensors")]

    if os.path.exists(pt_path):
        target_file = pt_path
        state_dict = torch.load(target_file, map_location="cpu", weights_only=True)
        is_safetensors = False
    elif safetensor_files:
        target_file = os.path.join(stage_dir, safetensor_files[0])
        state_dict = load_file(target_file)
        is_safetensors = True
    else:
        raise FileNotFoundError(f"No memory.pt or .safetensors found inside {stage_dir}")

    h_enc = 2560
    h_qwen = 4096
    h_gemma = 2816
    init_std = 1.0 / math.sqrt(h_enc)  # Standard fan-in std (~0.01976); forward hook applies sqrt(2816)

    torch.manual_seed(23)
    new_state_dict = {}
    preserved_params = 0
    resized_params = 0

    for key, tensor in state_dict.items():
        # Identify and resize the decoder output projection matrix
        if tensor.shape in [(h_qwen, h_enc), (h_enc, h_qwen)]:
            resized_weights = torch.randn(h_gemma, h_enc, dtype=tensor.dtype) * init_std
            new_tensor = (
                resized_weights if tensor.shape[0] == h_qwen else resized_weights.T.contiguous()
            )
            new_state_dict[key] = new_tensor
            resized_params += new_tensor.numel()
            actual_std = new_tensor.float().std().item()
            print(
                f"  [surgery] Resized projection matrix: {key} {tuple(tensor.shape)} -> {tuple(new_tensor.shape)} "
                f"| target_std={init_std:.5f}, actual_std={actual_std:.5f}"
            )
        elif tensor.shape == (h_qwen,):
            new_state_dict[key] = torch.zeros(h_gemma, dtype=tensor.dtype)
            resized_params += h_gemma
            print(f"  [surgery] Reset projection bias/norm vector: {key} {tuple(tensor.shape)} -> ({h_gemma},)")
        else:
            # Retain pretrained Perceiver cross-attention & latent slot parameters intact
            new_state_dict[key] = tensor.clone()
            preserved_params += tensor.numel()

    os.makedirs(output_dir, exist_ok=True)

    # Save atomically via .tmp -> os.replace so an interrupted write never corrupts memory.pt
    tmp_pt = out_pt + ".tmp"
    torch.save(new_state_dict, tmp_pt)
    os.replace(tmp_pt, out_pt)
    if is_safetensors:
        out_st = os.path.join(output_dir, safetensor_files[0])
        tmp_st = out_st + ".tmp"
        save_file(new_state_dict, tmp_st)
        os.replace(tmp_st, out_st)

    # Copy and update config.json metadata
    src_cfg = os.path.join(stage_dir, "config.json")
    if os.path.exists(src_cfg):
        with open(src_cfg, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        for dim_key in ["decoder_hidden_size", "output_dim", "llm_hidden_size", "d_model"]:
            if dim_key in cfg_data and cfg_data[dim_key] == h_qwen:
                cfg_data[dim_key] = h_gemma
        for dec_key in ["decoder", "decoder_model"]:
            if dec_key in cfg_data:
                cfg_data[dec_key] = "google/gemma-4-26B-A4B-it"
        cfg_data["gemma_embed_scale"] = math.sqrt(h_gemma)
        out_cfg = os.path.join(output_dir, "config.json")
        tmp_cfg = out_cfg + ".tmp"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        os.replace(tmp_cfg, out_cfg)
        print("  [config]  Updated module metadata: config.json")

    with open(complete_marker, "w", encoding="utf-8") as f:
        f.write(f"source_stage={source_stage}\nh_gemma={h_gemma}\n")

    size_mb = os.path.getsize(out_pt) / (1024 * 1024)
    elapsed_s = time.monotonic() - t0
    total_params = preserved_params + resized_params
    print(
        f"Structural conversion complete: {out_pt} ({size_mb:.2f} MB) | "
        f"Total Params: {total_params / 1e6:.2f}M (Preserved Perceiver: {preserved_params / 1e6:.2f}M, "
        f"Resized Projector: {resized_params / 1e6:.2f}M) | Elapsed: {elapsed_s:.2f}s"
    )


if __name__ == "__main__":
    perform_gemma4_surgery("stage2")
