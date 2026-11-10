# Save as subsample_data.py
import argparse
import os
import glob
import shutil
import subprocess
import time
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from memorilla.data import ensure_data

# Official dataset configs from docs/training.md
STAGE1_DATASETS = ["enwiki"]

STAGE2_DATASETS = [
    "squad_v2",
    "drop",
    "coqa",
    "pubmedqa",
    "quail",
    "pwc",
    "cnn_dailymail",
    "samsum",
    "dialogsum",
    "msmarco",
]

STAGE3_DATASETS = [
    "pv4",
    "pmv2",
    "factkg",
    "triviaqa",
    "narrativeqa",
    "pubmedqa",
    "lamp4",
    "lamp7",
]


def ensure_available_splits(dataset_name: str, candidate_splits=("train", "validation", "test")) -> str:
    """
    Downloads available splits and precomputed embeddings for `dataset_name` from memorilla/Memorilla-Data.
    Falls back to per-split fetching for datasets (e.g., enwiki) that omit validation/test splits.
    """
    cache_root = None
    try:
        cache_root = str(ensure_data(dataset_name, list(candidate_splits)))
    except Exception:
        for split in candidate_splits:
            try:
                cache_root = str(ensure_data(dataset_name, [split]))
            except Exception:
                continue
        if cache_root is None:
            raise RuntimeError(f"Could not download any splits for dataset '{dataset_name}'")

    # Ensure precomputed Qwen3-Embedding-4B vectors for this specific dataset are present in cache_root
    existing_ds_embs = glob.glob(os.path.join(cache_root, "embeddings", "*", dataset_name, "*", "index.safetensors"))
    if not existing_ds_embs:
        try:
            snapshot_download(
                repo_id="memorilla/Memorilla-Data",
                repo_type="dataset",
                allow_patterns=[
                    f"embeddings/*/{dataset_name}/*",
                    f"embeddings/*/{dataset_name}/*/*",
                ],
                local_dir=cache_root,
            )
        except Exception:
            pass
    return cache_root


def link_store_dir(src_dir: str, dst_dir: str) -> None:
    """
    Links precomputed embedding directories, falling back to an NTFS junction on Windows
    (or directory copy if filesystem links are disallowed).
    """
    src_abs = os.path.abspath(src_dir)
    dst_abs = os.path.abspath(dst_dir)
    try:
        os.symlink(src_abs, dst_abs, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            try:
                import _winapi

                _winapi.CreateJunction(src_abs, dst_abs)
                return
            except OSError:
                try:
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", dst_abs, src_abs],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except (OSError, subprocess.SubprocessError):
                    pass
        shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)


def link_or_copy_file(src_file: str, dst_file: str) -> None:
    src_abs = os.path.abspath(src_file)
    dst_abs = os.path.abspath(dst_file)
    if os.path.exists(dst_abs):
        return
    try:
        os.link(src_abs, dst_abs)
    except OSError:
        try:
            os.symlink(src_abs, dst_abs)
        except OSError:
            shutil.copy2(src_abs, dst_abs)


def compute_deterministic_train_indices(
    src_data_dir: str,
    fraction: float,
    min_train_rows: int = 2500,
    seed: int = 23,
) -> list[int]:
    """
    Computes the cumulative global row indices across all `train-*.parquet` shards
    selected by deterministic stratified sampling (`df.sample(..., random_state=seed).sort_index()`).
    """
    indices: list[int] = []
    row_offset = 0
    for pq_path in sorted(glob.glob(os.path.join(src_data_dir, "train*.parquet"))):
        table = pq.read_table(pq_path, columns=[])
        num_rows = table.num_rows
        df = pq.read_table(pq_path).to_pandas()
        target_rows = min(len(df), max(min_train_rows, int(len(df) * fraction)))
        df_sub = df.sample(n=target_rows, random_state=seed).sort_index()
        indices.extend(row_offset + int(i) for i in df_sub.index.to_list())
        row_offset += num_rows
    return indices


def sync_dataset_embeddings(
    dataset_name: str,
    cache_root: str,
    output_repo: str,
    train_selected_indices: list[int] | None,
) -> None:
    """
    Links precomputed document embedding shards and slices `train/questions.safetensors`
    to match the subsampled training split rows.
    """
    for emb_subdir in ["embeddings", "corpus", "collections"]:
        src_emb = os.path.join(cache_root, emb_subdir, dataset_name)
        dst_emb_parent = os.path.join(output_repo, emb_subdir)
        dst_emb = os.path.join(dst_emb_parent, dataset_name)
        if os.path.exists(src_emb) and not os.path.exists(dst_emb):
            os.makedirs(dst_emb_parent, exist_ok=True)
            link_store_dir(src_emb, dst_emb)
            print(f"  [link]  Linked precomputed store: {dst_emb} -> {src_emb}")

    for src_enc_ds in sorted(glob.glob(os.path.join(cache_root, "embeddings", "*", dataset_name))):
        encoder_slug = os.path.basename(os.path.dirname(src_enc_ds))
        dst_enc_ds = os.path.join(output_repo, "embeddings", encoder_slug, dataset_name)
        for split_dir in sorted(glob.glob(os.path.join(src_enc_ds, "*"))):
            if not os.path.isdir(split_dir):
                continue
            split_name = os.path.basename(split_dir)
            dst_split_dir = os.path.join(dst_enc_ds, split_name)
            q_src = os.path.join(split_dir, "questions.safetensors")
            if split_name == "train" and train_selected_indices is not None and os.path.exists(q_src):
                os.makedirs(dst_split_dir, exist_ok=True)
                q_dst = os.path.join(dst_split_dir, "questions.safetensors")
                needs_slice = True
                if os.path.exists(q_dst):
                    try:
                        with safe_open(q_dst, framework="pt") as h:
                            if int(h.get_slice("embeddings").get_shape()[0]) == len(train_selected_indices):
                                needs_slice = False
                    except Exception:
                        needs_slice = True
                if needs_slice:
                    q_tensors = load_file(q_src)
                    q_sub = {k: v[train_selected_indices].contiguous() for k, v in q_tensors.items()}
                    q_tmp = q_dst + ".tmp"
                    save_file(q_sub, q_tmp)
                    os.replace(q_tmp, q_dst)
                    print(
                        f"  [emb]   Subsampled train questions.safetensors ({len(train_selected_indices):,} rows) "
                        f"in {dst_split_dir}"
                    )
                for shard_file in os.listdir(split_dir):
                    if shard_file != "questions.safetensors":
                        link_or_copy_file(
                            os.path.join(split_dir, shard_file),
                            os.path.join(dst_split_dir, shard_file),
                        )
            elif not os.path.exists(dst_split_dir):
                os.makedirs(os.path.dirname(dst_split_dir), exist_ok=True)
                link_store_dir(split_dir, dst_split_dir)
                print(f"  [link]  Linked split embeddings: {dst_split_dir} -> {split_dir}")


def subsample_memorilla_config(
    dataset_name: str,
    fraction: float,
    output_repo: str,
    min_train_rows: int = 2500,
    seed: int = 23,
    force: bool = False,
    dataset_idx: int = 1,
    total_datasets: int = 1,
) -> dict:
    t0 = time.monotonic()
    dst_data_dir = os.path.join(output_repo, "data", dataset_name)
    complete_marker = os.path.join(dst_data_dir, ".complete")
    force_run = force or (os.environ.get("MEMORILLA_FORCE_SUBSAMPLE", "0") == "1")
    progress_prefix = f"[{dataset_idx}/{total_datasets} | {(dataset_idx / max(1, total_datasets)) * 100:5.1f}%]"

    if not force_run and os.path.exists(complete_marker):
        existing_parquets = glob.glob(os.path.join(dst_data_dir, "*.parquet"))
        if existing_parquets:
            # Verify that embeddings are also present and row-aligned before skipping
            expected_train_rows = sum(
                pq.read_metadata(p).num_rows
                for p in sorted(glob.glob(os.path.join(dst_data_dir, "train*.parquet")))
            )
            q_files = glob.glob(os.path.join(output_repo, "embeddings", "*", dataset_name, "train", "questions.safetensors"))
            emb_aligned = False
            if q_files:
                try:
                    with safe_open(q_files[0], framework="pt") as h:
                        emb_aligned = int(h.get_slice("embeddings").get_shape()[0]) == expected_train_rows
                except Exception:
                    emb_aligned = False
            if not emb_aligned:
                cache_root = str(ensure_available_splits(dataset_name, ("train", "validation", "test")))
                src_data_dir = os.path.join(cache_root, "data", dataset_name)
                if glob.glob(os.path.join(cache_root, "embeddings", "*", dataset_name)):
                    indices = compute_deterministic_train_indices(src_data_dir, fraction, min_train_rows, seed)
                    sync_dataset_embeddings(dataset_name, cache_root, output_repo, indices)
            print(
                f"=== {progress_prefix} [Resume] Skipping already completed dataset: "
                f"{dataset_name} ({len(existing_parquets)} splits) ==="
            )
            return {"status": "skipped", "train_rows": 0, "eval_rows": 0, "bytes_written": 0, "elapsed_s": 0.0}

    print(f"=== {progress_prefix} Processing Memorilla Dataset Config: {dataset_name} ===")
    # 1. Ensure official data + precomputed embeddings are downloaded to local cache
    cache_root = str(ensure_available_splits(dataset_name, ("train", "validation", "test")))

    src_data_dir = os.path.join(cache_root, "data", dataset_name)
    os.makedirs(dst_data_dir, exist_ok=True)

    # 2. Subsample train Parquet files (with a floor for small Stage 3 splits) while keeping eval splits intact
    parquet_files = sorted(glob.glob(os.path.join(src_data_dir, "*.parquet")))
    train_rows_total = 0
    eval_rows_total = 0
    bytes_written = 0
    train_selected_indices: list[int] = []
    train_row_offset = 0

    for pq_idx, pq_path in enumerate(parquet_files, start=1):
        split_t0 = time.monotonic()
        filename = os.path.basename(pq_path)
        dst_pq_path = os.path.join(dst_data_dir, filename)
        tmp_pq_path = dst_pq_path + ".tmp"
        table = pq.read_table(pq_path)

        if filename.startswith("train"):
            df = table.to_pandas()
            target_rows = min(len(df), max(min_train_rows, int(len(df) * fraction)))
            df_sub = df.sample(n=target_rows, random_state=seed).sort_index()
            train_selected_indices.extend(train_row_offset + int(i) for i in df_sub.index.to_list())
            train_row_offset += len(df)
            pq.write_table(
                pa.Table.from_pandas(df_sub, preserve_index=False),
                tmp_pq_path,
            )
            os.replace(tmp_pq_path, dst_pq_path)
            file_bytes = os.path.getsize(dst_pq_path)
            bytes_written += file_bytes
            train_rows_total += target_rows
            split_dt = max(1e-6, time.monotonic() - split_t0)
            pct = (target_rows / max(1, len(df))) * 100
            print(
                f"  [train {pq_idx}/{len(parquet_files)}] Subsampled {filename}: "
                f"{len(df):,} -> {target_rows:,} rows ({pct:.1f}%) | "
                f"{file_bytes / (1024 * 1024):.2f} MB | {target_rows / split_dt:,.0f} rows/s"
            )
        else:
            # Preserve full validation and test splits for Stage 3 epoch selection & evaluation
            pq.write_table(table, tmp_pq_path)
            os.replace(tmp_pq_path, dst_pq_path)
            file_bytes = os.path.getsize(dst_pq_path)
            bytes_written += file_bytes
            eval_rows_total += table.num_rows
            split_dt = max(1e-6, time.monotonic() - split_t0)
            print(
                f"  [eval  {pq_idx}/{len(parquet_files)}] Copied full split {filename}: "
                f"{table.num_rows:,} rows | {file_bytes / (1024 * 1024):.2f} MB | "
                f"{table.num_rows / split_dt:,.0f} rows/s"
            )

    # 3. Link precomputed Qwen3-Embedding-4B vectors (and slice row-aligned train questions.safetensors)
    sync_dataset_embeddings(
        dataset_name,
        cache_root,
        output_repo,
        train_selected_indices if train_selected_indices else None,
    )

    with open(complete_marker, "w", encoding="utf-8") as f:
        f.write(f"fraction={fraction}\nmin_train_rows={min_train_rows}\nseed={seed}\n")

    elapsed_s = time.monotonic() - t0
    print(
        f"Successfully prepared {dataset_name} in {output_repo} "
        f"[train={train_rows_total:,} rows, eval={eval_rows_total:,} rows, "
        f"disk={bytes_written / (1024 * 1024):.2f} MB, elapsed={elapsed_s:.2f}s]\n"
    )
    return {
        "status": "processed",
        "train_rows": train_rows_total,
        "eval_rows": eval_rows_total,
        "bytes_written": bytes_written,
        "elapsed_s": elapsed_s,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample Memorilla datasets and align precomputed embeddings.")
    parser.add_argument("--data_dir", "--output_dir", dest="output_dir", type=str, default="./data_subsampled")
    parser.add_argument("--ratio", "--fraction", dest="fraction", type=float, default=0.10)
    parser.add_argument("--min_train_rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--force", action="store_true")
    cli_args = parser.parse_args()

    OUTPUT_REPO = cli_args.output_dir
    os.makedirs(OUTPUT_REPO, exist_ok=True)

    # Deduplicate across stages (e.g., pubmedqa appears in both Stage 2 and Stage 3)
    all_datasets = list(dict.fromkeys(STAGE1_DATASETS + STAGE2_DATASETS + STAGE3_DATASETS))
    suite_t0 = time.monotonic()
    processed_count = 0
    skipped_count = 0
    total_train = 0
    total_eval = 0

    for idx, ds in enumerate(all_datasets, start=1):
        stats = subsample_memorilla_config(
            ds,
            fraction=cli_args.fraction,
            output_repo=OUTPUT_REPO,
            min_train_rows=cli_args.min_train_rows,
            seed=cli_args.seed,
            force=cli_args.force,
            dataset_idx=idx,
            total_datasets=len(all_datasets),
        )
        if stats["status"] == "skipped":
            skipped_count += 1
        else:
            processed_count += 1
            total_train += stats["train_rows"]
            total_eval += stats["eval_rows"]

    suite_dt = time.monotonic() - suite_t0
    print(
        f"=== Subsampling Complete: {processed_count} processed, {skipped_count} skipped (resumed) | "
        f"Total Train Rows: {total_train:,} | Total Eval Rows: {total_eval:,} | "
        f"Elapsed: {suite_dt:.1f}s ==="
    )
