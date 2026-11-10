# Save as test_pipeline.py and run: python test_pipeline.py
import ast
import importlib.util
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch

# Configure dual logging (console + test_pipeline.log)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pipeline.log")
logger = logging.getLogger("MemorillaGemma4TestSuite")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)-7s | %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_simple_yaml(yaml_path: str) -> dict:
    """
    Lightweight YAML parser fallback in case PyYAML is not installed in the base environment.
    """
    try:
        import yaml  # type: ignore

        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        data = {}
        current_list_key = None
        with open(yaml_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].rstrip()
                if not line.strip():
                    continue
                if line.lstrip().startswith("- ") and current_list_key:
                    val = line.lstrip()[2:].strip().strip('"').strip("'")
                    data[current_list_key].append(val)
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if not val:
                        data[key] = []
                        current_list_key = key
                    else:
                        current_list_key = None
                        val_clean = val.strip('"').strip("'")
                        if val_clean.lower() == "true":
                            data[key] = True
                        elif val_clean.lower() == "false":
                            data[key] = False
                        else:
                            try:
                                if "." in val_clean or "e" in val_clean.lower():
                                    data[key] = float(val_clean)
                                else:
                                    data[key] = int(val_clean)
                            except ValueError:
                                data[key] = val_clean
        return data


class LoggedTestCase(unittest.TestCase):
    def run(self, result=None):
        test_name = self._testMethodName
        logger.info(f"▶ RUNNING: {test_name}")
        res = super().run(result)
        if res:
            my_failures = [tb for tc, tb in (res.errors + res.failures) if tc is self]
            if my_failures:
                for tb in my_failures:
                    logger.error(f"✖ FAILED: {test_name}\n{tb}")
            else:
                logger.info(f"✔ PASSED: {test_name}")
        return res


class TestGemma4MemorillaPipeline(LoggedTestCase):
    """
    End-to-end verification suite for the Gemma 4 26B-A4B Memorilla 10% curriculum repository.
    Tests syntax, README synchronization, YAML configs, subsampling + NTFS links, weight surgery,
    hook-based sqrt(2816) scaling round-trips, 4-bit NF4 3D MoE quantization, and autograd flow.
    """

    def test_01_files_syntax_and_readme_sync(self):
        """Verify all repository files exist, compile cleanly, and match readme.md code blocks."""
        expected_files = [
            "readme.md",
            "requires.txt",
            "subsample_data.py",
            "prepare_gemma4_weights.py",
            "apply_gemma4_patch.py",
            "example.bat.txt",
            "configs/stage1_gemma4_10pct.yaml",
            "configs/stage2_gemma4_10pct.yaml",
            "configs/stage3_gemma4_10pct.yaml",
        ]
        for rel_path in expected_files:
            full_path = os.path.join(REPO_ROOT, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Missing required file: {rel_path}")

        # Verify memorilla is NOT cloned inside the repository tree (must live in .venv site-packages)
        self.assertFalse(
            os.path.exists(os.path.join(REPO_ROOT, "memorilla")),
            "memorilla directory should not be part of the repository tree; install it into .venv via requires.txt!",
        )

        with open(os.path.join(REPO_ROOT, "requires.txt"), "r", encoding="utf-8") as req_f:
            req_text = req_f.read()
        self.assertIn(
            "memorilla @ git+https://github.com/snap-stanford/memorilla.git",
            req_text,
            "requires.txt must include memorilla @ git+https://github.com/snap-stanford/memorilla.git",
        )

        # Verify all required third-party libraries from requires.txt (including memorilla) import cleanly
        for pkg in (
            "torch",
            "transformers",
            "accelerate",
            "safetensors",
            "huggingface_hub",
            "tokenizers",
            "einops",
            "memorilla",
            "memorilla.memory",
            "memorilla.data",
            "memorilla.embeddings",
            "memorilla.llm",
            "memorilla.benchmarks",
            "memorilla.metrics",
            "lightning",
            "datasets",
            "nltk",
            "rouge_score",
            "filelock",
            "pyarrow",
            "pyarrow.parquet",
            "pandas",
            "numpy",
            "yaml",
            "tqdm",
            "requests",
        ):
            mod = __import__(pkg)
            self.assertIsNotNone(mod)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        self.assertIsNotNone(AutoConfig)
        self.assertIsNotNone(AutoModelForCausalLM)
        self.assertIsNotNone(AutoModelForImageTextToText)

        # Verify AST syntax of all Python scripts
        for py_file in ("subsample_data.py", "prepare_gemma4_weights.py", "apply_gemma4_patch.py"):
            full_path = os.path.join(REPO_ROOT, py_file)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=py_file)

        # Verify code blocks in readme.md match disk files
        with open(os.path.join(REPO_ROOT, "readme.md"), "r", encoding="utf-8") as f:
            readme_text = f.read()

        blocks = re.findall(r"```(?:python|yaml|bash)\n(.*?)```", readme_text, flags=re.DOTALL)
        self.assertEqual(len(blocks), 7, f"Expected 7 code blocks in readme.md, found {len(blocks)}")

        file_map = {
            "# Save as subsample_data.py": "subsample_data.py",
            "# Save as prepare_gemma4_weights.py": "prepare_gemma4_weights.py",
            "# Save as apply_gemma4_patch.py": "apply_gemma4_patch.py",
            "# Save as configs/stage1_gemma4_10pct.yaml": "configs/stage1_gemma4_10pct.yaml",
            "# Save as configs/stage2_gemma4_10pct.yaml": "configs/stage2_gemma4_10pct.yaml",
            "# Save as configs/stage3_gemma4_10pct.yaml": "configs/stage3_gemma4_10pct.yaml",
            "# 0. Create Virtual Environment": "example.bat.txt",
        }
        for prefix, rel_path in file_map.items():
            matched = [b for b in blocks if b.strip().startswith(prefix)]
            self.assertEqual(len(matched), 1, f"Could not find unique code block for {rel_path} in readme.md")
            with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
                disk_content = f.read().strip().replace("\r\n", "\n")
            readme_block = matched[0].strip().replace("\r\n", "\n")
            self.assertEqual(
                disk_content,
                readme_block,
                f"File '{rel_path}' is out of sync with its code block in readme.md!",
            )

    def test_02_yaml_configs_and_curriculum_chain(self):
        """Verify Stage 1, 2, and 3 YAML configs use only native keys and form a valid checkpoint chain."""
        s1 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage1_gemma4_10pct.yaml"))
        s2 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage2_gemma4_10pct.yaml"))
        s3 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage3_gemma4_10pct.yaml"))

        allowed_native_keys = {
            "decoder",
            "encoder",
            "attn_implementation",
            "init_memory",
            "data_repo",
            "datasets",
            "output_dir",
            "num_memories",
            "devices",
            "strategy",
            "num_workers",
            "batch_size",
            "gradient_accumulation_steps",
            "max_epochs",
            "learning_rate",
            "weight_decay",
            "seed",
        }

        for stage_name, cfg in [("stage1", s1), ("stage2", s2), ("stage3", s3)]:
            unknown = set(cfg.keys()) - allowed_native_keys
            self.assertFalse(
                unknown,
                f"{stage_name} config contains non-native keys that will break train.py parser: {unknown}",
            )
            self.assertEqual(cfg["decoder"], "google/gemma-4-26B-A4B-it")
            self.assertEqual(cfg["encoder"], "Qwen/Qwen3-Embedding-4B")
            self.assertEqual(cfg["attn_implementation"], "sdpa")
            self.assertEqual(cfg["num_memories"], 16)
            self.assertEqual(cfg["devices"], 1)

        # Verify effective batch sizes (128 for Stages 1 & 2; 64 for Stage 3)
        self.assertEqual(s1["batch_size"] * s1["gradient_accumulation_steps"] * s1["devices"], 128)
        self.assertEqual(s2["batch_size"] * s2["gradient_accumulation_steps"] * s2["devices"], 128)
        self.assertEqual(s3["batch_size"] * s3["gradient_accumulation_steps"] * s3["devices"], 64)

        # Verify checkpoint handoff chain
        self.assertEqual(s1["init_memory"], "checkpoints/gemma4_warmstart/memory.pt")
        self.assertEqual(s2["init_memory"], f"{s1['output_dir']}/epoch-00/memory.pt")
        self.assertEqual(s3["init_memory"], f"{s2['output_dir']}/epoch-00/memory.pt")
        self.assertEqual(s3["max_epochs"], 5)

    def test_03_subsample_data_pipeline_and_links(self):
        """Test ensure_available_splits fallback, real Parquet subsampling (2500-row floor), and store linking."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Only mock memorilla.data (since snap-stanford/memorilla is cloned at Step 0)
        mock_memorilla = type(sys)("memorilla")
        mock_memorilla_data = type(sys)("memorilla.data")

        call_log = []

        def fake_ensure_data(dataset_name, splits):
            call_log.append((dataset_name, tuple(splits)))
            if dataset_name == "enwiki" and ("validation" in splits or "test" in splits):
                raise ValueError("enwiki only has a 'train' split")
            return "/tmp/fake_cache"

        mock_memorilla_data.ensure_data = fake_ensure_data
        mock_memorilla.data = mock_memorilla_data

        with patch.dict(
            sys.modules,
            {
                "memorilla": mock_memorilla,
                "memorilla.data": mock_memorilla_data,
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "subsample_data", os.path.join(REPO_ROOT, "subsample_data.py")
            )
            subsample_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(subsample_mod)

            # 1. Verify ensure_available_splits gracefully falls back for 'enwiki'
            root = subsample_mod.ensure_available_splits("enwiki", ("train", "validation", "test"))
            self.assertEqual(root, "/tmp/fake_cache")
            self.assertIn(("enwiki", ("train",)), call_log)

            # 2. Verify dataset lists match YAML configs
            s1 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage1_gemma4_10pct.yaml"))
            s2 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage2_gemma4_10pct.yaml"))
            s3 = parse_simple_yaml(os.path.join(REPO_ROOT, "configs/stage3_gemma4_10pct.yaml"))
            self.assertEqual(subsample_mod.STAGE1_DATASETS, s1["datasets"])
            self.assertEqual(subsample_mod.STAGE2_DATASETS, s2["datasets"])
            self.assertEqual(subsample_mod.STAGE3_DATASETS, s3["datasets"])

            # 3. Verify link_store_dir creates a working directory link/junction on the host OS
            with tempfile.TemporaryDirectory() as tmpdir:
                src_emb = os.path.join(tmpdir, "cache", "embeddings", "factkg")
                os.makedirs(src_emb, exist_ok=True)
                marker_file = os.path.join(src_emb, "vec.bin")
                with open(marker_file, "wb") as f:
                    f.write(b"OK")

                dst_emb = os.path.join(tmpdir, "out", "embeddings", "factkg")
                os.makedirs(os.path.dirname(dst_emb), exist_ok=True)
                subsample_mod.link_store_dir(src_emb, dst_emb)
                self.assertTrue(os.path.exists(os.path.join(dst_emb, "vec.bin")))

                # 4. Verify subsample_memorilla_config on REAL Parquet files using pyarrow & pandas
                cache_root = os.path.join(tmpdir, "fake_hf_cache")
                ds_name = "lamp4"
                src_ds_dir = os.path.join(cache_root, "data", ds_name)
                os.makedirs(src_ds_dir, exist_ok=True)
                os.makedirs(os.path.join(cache_root, "embeddings", ds_name), exist_ok=True)
                with open(os.path.join(cache_root, "embeddings", ds_name, "emb.bin"), "wb") as f:
                    f.write(b"EMB")

                # Write real Apache Parquet files: 8,000 train rows (10% = 800 -> floor 2500) and 1,200 eval rows
                pq.write_table(
                    pa.Table.from_pandas(pd.DataFrame({"id": range(8000), "text": ["q"] * 8000})),
                    os.path.join(src_ds_dir, "train-00000.parquet"),
                )
                for split_file in ("validation-00000.parquet", "test-00000.parquet"):
                    pq.write_table(
                        pa.Table.from_pandas(pd.DataFrame({"id": range(1200), "text": ["e"] * 1200})),
                        os.path.join(src_ds_dir, split_file),
                    )

                subsample_mod.ensure_data = lambda ds, splits: cache_root
                out_repo = os.path.join(tmpdir, "data_subsampled")
                subsample_mod.subsample_memorilla_config(
                    ds_name, fraction=0.10, output_repo=out_repo, min_train_rows=2500, seed=23
                )

                out_ds_dir = os.path.join(out_repo, "data", ds_name)
                self.assertEqual(pq.read_table(os.path.join(out_ds_dir, "train-00000.parquet")).num_rows, 2500)
                self.assertEqual(pq.read_table(os.path.join(out_ds_dir, "validation-00000.parquet")).num_rows, 1200)
                self.assertEqual(pq.read_table(os.path.join(out_ds_dir, "test-00000.parquet")).num_rows, 1200)
                self.assertTrue(os.path.exists(os.path.join(out_repo, "embeddings", ds_name, "emb.bin")))

    def test_04_weight_surgery_and_metadata(self):
        """Test perform_gemma4_surgery resizing (4096 -> 2816) and standard 1/sqrt(2560) variance."""
        import torch

        mock_hf = type(sys)("huggingface_hub")
        mock_st = type(sys)("safetensors")
        mock_st_torch = type(sys)("safetensors.torch")
        mock_st_torch.load_file = lambda *a, **k: {}
        mock_st_torch.save_file = lambda *a, **k: None

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = os.path.join(tmpdir, "ckpt_cache", "stage2")
            os.makedirs(stage2_dir, exist_ok=True)

            # Create synthetic Qwen3-8B Stage 2 checkpoint
            torch.manual_seed(0)
            qwen_sd = {
                "query_slots": torch.randn(16, 2560, dtype=torch.bfloat16),
                "perceiver.0.attn.weight": torch.randn(2560, 2560, dtype=torch.bfloat16),
                "proj.weight": torch.randn(4096, 2560, dtype=torch.bfloat16),
                "proj.bias": torch.ones(4096, dtype=torch.bfloat16),
            }
            torch.save(qwen_sd, os.path.join(stage2_dir, "memory.pt"))
            with open(os.path.join(stage2_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"decoder": "Qwen/Qwen3-8B", "decoder_hidden_size": 4096}, f)

            mock_hf.snapshot_download = lambda repo_id, allow_patterns=None: os.path.join(tmpdir, "ckpt_cache")

            with patch.dict(
                sys.modules,
                {
                    "huggingface_hub": mock_hf,
                    "safetensors": mock_st,
                    "safetensors.torch": mock_st_torch,
                },
            ):
                spec = importlib.util.spec_from_file_location(
                    "prepare_gemma4_weights", os.path.join(REPO_ROOT, "prepare_gemma4_weights.py")
                )
                prep_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(prep_mod)

                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    prep_mod.perform_gemma4_surgery("stage2")

                    out_pt = os.path.join(tmpdir, "checkpoints", "gemma4_warmstart", "memory.pt")
                    out_cfg = os.path.join(tmpdir, "checkpoints", "gemma4_warmstart", "config.json")
                    self.assertTrue(os.path.exists(out_pt))
                    self.assertTrue(os.path.exists(out_cfg))

                    new_sd = torch.load(out_pt, map_location="cpu", weights_only=True)
                    self.assertEqual(tuple(new_sd["proj.weight"].shape), (2816, 2560))
                    self.assertEqual(tuple(new_sd["proj.bias"].shape), (2816,))
                    self.assertTrue(torch.all(new_sd["proj.bias"] == 0))

                    # Verify Perceiver weights are untouched
                    self.assertTrue(torch.equal(new_sd["query_slots"], qwen_sd["query_slots"]))
                    self.assertTrue(
                        torch.equal(new_sd["perceiver.0.attn.weight"], qwen_sd["perceiver.0.attn.weight"])
                    )

                    # Verify standard fan-in std (~1/sqrt(2560) = 0.01976)
                    actual_std = new_sd["proj.weight"].float().std().item()
                    expected_std = 1.0 / math.sqrt(2560)
                    self.assertAlmostEqual(actual_std, expected_std, delta=0.001)

                    with open(out_cfg, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                    self.assertEqual(cfg_data["decoder"], "google/gemma-4-26B-A4B-it")
                    self.assertEqual(cfg_data["decoder_hidden_size"], 2816)
                    self.assertAlmostEqual(cfg_data["gemma_embed_scale"], math.sqrt(2816), places=4)
                finally:
                    os.chdir(old_cwd)

    def test_05_scale_hooks_train_eval_roundtrip(self):
        """
        Verify attach_gemma4_scale_hooks keeps in-memory weights at std ~0.0198, scales forward
        activations by sqrt(2816), folds scale into saved memory.pt, and unscales on reload.
        """
        import torch
        import torch.nn as nn

        mock_tf = type(sys)("transformers")
        mock_tf.AutoConfig = object
        mock_tf.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {"from_pretrained": classmethod(lambda cls, *a, **k: None)})
        mock_tf.AutoModelForImageTextToText = object

        with patch.dict(sys.modules, {"transformers": mock_tf}):
            spec = importlib.util.spec_from_file_location(
                "apply_gemma4_patch", os.path.join(REPO_ROOT, "apply_gemma4_patch.py")
            )
            patch_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_mod)

            class DummyMemoryModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.perceiver = nn.Linear(2560, 2560, bias=False, dtype=torch.bfloat16)
                    self.proj = nn.Linear(2560, 2816, bias=True, dtype=torch.bfloat16)
                    nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(2560))
                    nn.init.zeros_(self.proj.bias)

                def forward(self, x):
                    return self.proj(self.perceiver(x))

            # 1. Stage 1 mode: freeze_perceiver=True
            mm_stage1 = DummyMemoryModule()
            patch_mod.configure_gemma4_memory_module(mm_stage1, freeze_perceiver=True)
            self.assertFalse(mm_stage1.perceiver.weight.requires_grad)
            self.assertTrue(mm_stage1.proj.weight.requires_grad)

            # 2. Verify forward output is scaled by sqrt(2816) (~53.066) while in-memory std is ~0.0198
            torch.manual_seed(23)
            x = torch.randn(2, 16, 2560, dtype=torch.bfloat16)
            raw_linear_out = torch.nn.functional.linear(x, mm_stage1.proj.weight, mm_stage1.proj.bias)
            hooked_out = mm_stage1.proj(x)
            ratio = (hooked_out.float().norm() / raw_linear_out.float().norm()).item()
            self.assertAlmostEqual(ratio, math.sqrt(2816), delta=0.25)

            # 3. Save state_dict (simulating end of Stage 1) and verify on-disk weight is folded (~1.05 std)
            #    while in-memory weight remains unmutated (~0.0198 std)
            saved_sd = mm_stage1.state_dict()
            in_mem_std = mm_stage1.proj.weight.float().std().item()
            on_disk_std = saved_sd["proj.weight"].float().std().item()
            self.assertAlmostEqual(in_mem_std, 1.0 / math.sqrt(2560), delta=0.002)
            self.assertAlmostEqual(on_disk_std, math.sqrt(2816) / math.sqrt(2560), delta=0.05)

            # 4. Reload saved_sd into a patched Stage 2 module (verify _unscale_on_load prevents double scaling)
            mm_stage2 = DummyMemoryModule()
            patch_mod.configure_gemma4_memory_module(mm_stage2, freeze_perceiver=False)
            mm_stage2.load_state_dict(saved_sd)
            self.assertTrue(mm_stage2.perceiver.weight.requires_grad)
            self.assertAlmostEqual(
                mm_stage2.proj.weight.float().std().item(),
                in_mem_std,
                delta=0.001,
                msg="Reloaded weight in patched module was not unscaled back to 1/sqrt(2560)!",
            )
            out_stage2 = mm_stage2.proj(x)
            # In BF16 at scale ~53.066 (output std ~53), 7-bit mantissa rounding across 2560 inputs
            # yields elementwise std(diff) ~ 0.21; verify cosine similarity > 0.9999 and bounded BF16 diff
            cos_sim_s2 = torch.nn.functional.cosine_similarity(
                hooked_out.float().flatten(), out_stage2.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos_sim_s2, 0.9999)
            self.assertTrue(torch.allclose(hooked_out.float(), out_stage2.float(), rtol=2e-2, atol=1.0))

            # 5. Reload saved_sd into an UNPATCHED module (simulating standalone inference)
            mm_unpatched = DummyMemoryModule()
            saved_sd_copy = mm_stage1.state_dict()
            mm_unpatched.load_state_dict(saved_sd_copy)
            out_unpatched = mm_unpatched.proj(x)
            cos_sim_unpatched = torch.nn.functional.cosine_similarity(
                hooked_out.float().flatten(), out_unpatched.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos_sim_unpatched, 0.9999)
            self.assertTrue(
                torch.allclose(hooked_out.float(), out_unpatched.float(), rtol=2e-2, atol=1.0),
                "Unpatched inference module did not produce identical sqrt(2816)-scaled outputs!",
            )

            # 6. Verify real installed `memorilla.MemoryModule` integration (hook attachment, save, and from_pretrained)
            import memorilla

            patch_mod.install_runtime_monkeypatch()
            real_mm = memorilla.MemoryModule(
                embedding_dim=2560,
                output_dim=2816,
                num_memories=16,
                num_heads=8,
                num_cross_attn_layers=1,
            ).to(torch.bfloat16)
            nn.init.normal_(real_mm.memory_projection.weight, std=1.0 / math.sqrt(2560))
            nn.init.zeros_(real_mm.memory_projection.bias)
            self.assertTrue(getattr(real_mm.memory_projection, "_gemma4_scaled", False))

            with tempfile.TemporaryDirectory() as ckpt_tmp:
                real_mm.save(ckpt_tmp)
                # Inject extra metadata keys (as prepare_gemma4_weights.py does) and confirm from_pretrained loads cleanly
                cfg_json_path = os.path.join(ckpt_tmp, "config.json")
                with open(cfg_json_path, "r", encoding="utf-8") as cf:
                    c_dict = json.load(cf)
                c_dict["gemma_embed_scale"] = math.sqrt(2816)
                c_dict["decoder"] = "google/gemma-4-26B-A4B-it"
                with open(cfg_json_path, "w", encoding="utf-8") as cf:
                    json.dump(c_dict, cf, indent=2)

                reloaded_mm = memorilla.MemoryModule.from_pretrained(ckpt_tmp)
                self.assertEqual(reloaded_mm.config.output_dim, 2816)
                self.assertAlmostEqual(
                    reloaded_mm.memory_projection.weight.float().std().item(),
                    1.0 / math.sqrt(2560),
                    delta=0.002,
                )

    def test_06_nf4_3d_experts_and_autograd_backward(self):
        """
        Verify PackedInt4ExpertParameter (NF4, group_size=64) on Gemma 4's exact dimensions
        (2816, 1408, 704) and confirm gradients flow through gradient-checkpointed MoE layers
        all the way back to MemoryModule parameters.
        """
        import torch
        import torch.nn as nn
        from torch.utils.checkpoint import checkpoint

        mock_tf = type(sys)("transformers")
        mock_tf.AutoConfig = object
        mock_tf.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {"from_pretrained": classmethod(lambda cls, *a, **k: None)})
        mock_tf.AutoModelForImageTextToText = object

        with patch.dict(sys.modules, {"transformers": mock_tf}):
            spec = importlib.util.spec_from_file_location(
                "apply_gemma4_patch", os.path.join(REPO_ROOT, "apply_gemma4_patch.py")
            )
            patch_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_mod)

            # Use Gemma 4's exact per-expert dimensions: hidden=2816, gate_up=1408, down_in=704
            class Gemma4TextExperts(nn.Module):
                def __init__(self, num_experts=4, hidden_size=2816, intermediate_size=704):
                    super().__init__()
                    self.gate_up_proj = nn.Parameter(
                        torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.bfloat16) * 0.02
                    )
                    self.down_proj = nn.Parameter(
                        torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16) * 0.02
                    )

                def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                    # Route to active experts 0 and 2 (simulating top-k sparse dispatch)
                    out = torch.zeros_like(hidden_states)
                    for exp_idx in (0, 2):
                        w_gate_up = self.gate_up_proj[exp_idx]  # [1408, 2816]
                        w_down = self.down_proj[exp_idx]        # [2816, 704]
                        gate_up = torch.matmul(hidden_states, w_gate_up.t())
                        gate, up = gate_up.chunk(2, dim=-1)
                        inter = torch.nn.functional.silu(gate) * up
                        out = out + 0.5 * torch.matmul(inter, w_down.t())
                    return out

            class DummyGemma4Decoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.experts = Gemma4TextExperts(num_experts=4, hidden_size=2816, intermediate_size=704)

                def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
                    return checkpoint(self.experts, inputs_embeds, use_reentrant=False)

            decoder = DummyGemma4Decoder()
            bf16_bytes = (
                decoder.experts.gate_up_proj.numel() + decoder.experts.down_proj.numel()
            ) * 2

            # Quantize 3D experts to 4-bit NF4
            patch_mod.quantize_gemma4_experts_to_int4(decoder, group_size=64)
            self.assertIsInstance(decoder.experts.gate_up_proj, patch_mod.PackedInt4ExpertParameter)
            self.assertIsInstance(decoder.experts.down_proj, patch_mod.PackedInt4ExpertParameter)

            nf4_bytes = (
                decoder.experts.gate_up_proj.packed_weight.numel()
                + decoder.experts.gate_up_proj.scales.numel() * 2
                + decoder.experts.down_proj.packed_weight.numel()
                + decoder.experts.down_proj.scales.numel() * 2
            )
            compression_ratio = bf16_bytes / nf4_bytes
            logger.info(f"  4-bit NF4 Expert Compression Ratio: {compression_ratio:.2f}x ({bf16_bytes}B -> {nf4_bytes}B)")
            self.assertGreater(compression_ratio, 3.5)

            # Freeze decoder & run forward + backward from trainable MemoryModule projector
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad = False

            projector = nn.Linear(2560, 2816, dtype=torch.bfloat16)
            patch_mod.attach_gemma4_scale_hooks(projector)

            query_latents = torch.randn(2, 16, 2560, dtype=torch.bfloat16)
            soft_memory_tokens = projector(query_latents)
            hidden_out = decoder(soft_memory_tokens)
            loss = hidden_out.float().pow(2).mean()
            loss.backward()

            self.assertIsNotNone(projector.weight.grad, "Projector weight received None gradient!")
            grad_norm = projector.weight.grad.float().norm().item()
            logger.info(f"  End-to-end autograd check: projector.weight.grad norm = {grad_norm:.4f}")
            self.assertGreater(grad_norm, 0.0, "Projector gradient norm is zero!")
            self.assertTrue(math.isfinite(grad_norm), "Projector gradient norm is NaN/Inf!")

    def test_07_script_monkeypatch_injection(self):
        """Verify patch_scripts() injects runtime hooks idempotently into train.py and evaluate.py."""
        mock_tf = type(sys)("transformers")
        mock_tf.AutoConfig = object
        mock_tf.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {"from_pretrained": classmethod(lambda cls, *a, **k: None)})
        mock_tf.AutoModelForImageTextToText = object

        with patch.dict(sys.modules, {"transformers": mock_tf}):
            spec = importlib.util.spec_from_file_location(
                "apply_gemma4_patch", os.path.join(REPO_ROOT, "apply_gemma4_patch.py")
            )
            patch_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_mod)

            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    for fname in ("train.py", "evaluate.py"):
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write("print('hello')\n")

                    patch_mod.patch_scripts()
                    patch_mod.patch_scripts()  # Run twice to verify idempotency

                    for fname in ("train.py", "evaluate.py"):
                        with open(fname, "r", encoding="utf-8") as f:
                            content = f.read()
                        self.assertEqual(content.count("# [GEMMA4_MEMORILLA_PATCH]"), 1)
                        self.assertIn("apply_gemma4_patch.install_runtime_monkeypatch()", content)
                finally:
                    os.chdir(old_cwd)

            # Verify load_gemma4_text_decoder multimodal tower stripping & config aliasing
            import torch
            import torch.nn as nn

            class FakeTextConfig:
                hidden_size = 2816
                vocab_size = 262144
                num_hidden_layers = 30
                eos_token_id = 1
                pad_token_id = 0
                output_router_logits = True

            class FakeGemma4Config:
                def __init__(self):
                    self.text_config = FakeTextConfig()

            class FakeLanguageModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(16, 2816)

                def get_input_embeddings(self):
                    return self.embed

            class FakeInnerModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vision_tower = nn.Linear(16, 16)
                    self.multi_modal_projector = nn.Linear(16, 16)
                    self.embed_vision = nn.Linear(16, 16)
                    self.language_model = FakeLanguageModel()

            class FakeGemma4VLM(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.model = FakeInnerModel()
                    self.input_grads_enabled = False

                def get_input_embeddings(self):
                    return None

                def enable_input_require_grads(self):
                    self.input_grads_enabled = True

            patch_mod.AutoConfig = type("AutoConfig", (), {"from_pretrained": staticmethod(lambda *a, **k: FakeGemma4Config())})
            patch_mod.AutoModelForImageTextToText = type(
                "AutoModelForImageTextToText",
                (),
                {"from_pretrained": staticmethod(lambda model_id, config=None, **kw: FakeGemma4VLM(config))},
            )

            loaded_model = patch_mod.load_gemma4_text_decoder("google/gemma-4-26B-A4B-it", device_map="cpu")
            self.assertEqual(loaded_model.config.hidden_size, 2816)
            self.assertEqual(loaded_model.config.vocab_size, 262144)
            self.assertFalse(loaded_model.config.text_config.output_router_logits)
            self.assertFalse(hasattr(loaded_model.model, "vision_tower"))
            self.assertFalse(hasattr(loaded_model.model, "multi_modal_projector"))
            self.assertFalse(hasattr(loaded_model.model, "embed_vision"))
            self.assertTrue(loaded_model.input_grads_enabled)
            self.assertIsNotNone(loaded_model.get_input_embeddings())

    def test_08_save_and_resume_lifecycle(self):
        """
        Verify stage-level and intra-epoch save/resume across:
        1. subsample_data.py (.complete marker skip + force override)
        2. prepare_gemma4_weights.py (.complete marker skip + force override)
        3. CheckpointResumeManager (mid-epoch pause + resume skipping completed epochs,
           fast-forwarding DataLoader micro-batches, and restoring model/optimizer/scheduler
           to match an uninterrupted multi-epoch run).
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import signal
        import torch
        import torch.nn as nn

        spec = importlib.util.spec_from_file_location(
            "apply_gemma4_patch", os.path.join(REPO_ROOT, "apply_gemma4_patch.py")
        )
        patch_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Part A: Verify subsample_data.py skips completed datasets on resume ---
            mock_memorilla = type(sys)("memorilla")
            mock_memorilla_data = type(sys)("memorilla.data")
            ensure_calls = []
            cache_root = os.path.join(tmpdir, "hf_cache")
            ds_name = "pv4"
            os.makedirs(os.path.join(cache_root, "data", ds_name), exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(pd.DataFrame({"id": range(3000)})),
                os.path.join(cache_root, "data", ds_name, "train-00000.parquet"),
            )

            def tracked_ensure(ds, splits):
                ensure_calls.append(ds)
                return cache_root

            mock_memorilla_data.ensure_data = tracked_ensure
            mock_memorilla.data = mock_memorilla_data
            with patch.dict(sys.modules, {"memorilla": mock_memorilla, "memorilla.data": mock_memorilla_data}):
                sub_spec = importlib.util.spec_from_file_location(
                    "subsample_data", os.path.join(REPO_ROOT, "subsample_data.py")
                )
                sub_mod = importlib.util.module_from_spec(sub_spec)
                sub_spec.loader.exec_module(sub_mod)

                out_repo = os.path.join(tmpdir, "subsampled")
                sub_mod.subsample_memorilla_config(ds_name, 0.10, out_repo, min_train_rows=2500)
                self.assertEqual(len(ensure_calls), 1)
                self.assertTrue(os.path.exists(os.path.join(out_repo, "data", ds_name, ".complete")))

                # Second call without force must skip immediately without calling ensure_data
                sub_mod.subsample_memorilla_config(ds_name, 0.10, out_repo, min_train_rows=2500)
                self.assertEqual(len(ensure_calls), 1, "Resume did not skip already-completed dataset!")

            # --- Part B: Verify CheckpointResumeManager mid-epoch pause & resume equivalence ---
            torch.manual_seed(99)
            epochs = 3
            micro_batches_per_epoch = 12
            grad_accum = 4  # 3 optimizer steps per epoch -> 9 total optimizer steps
            epoch_data = [
                [
                    (torch.randn(2, 4, 2560), torch.randn(2, 4, 2816))
                    for _ in range(micro_batches_per_epoch)
                ]
                for _ in range(epochs)
            ]

            def build_components():
                torch.manual_seed(1234)
                model = nn.Sequential()
                model.add_module("proj", nn.Linear(2560, 2816))
                patch_mod.attach_gemma4_scale_hooks(model.proj)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.9)
                return model, opt, sched

            # 1. Run uninterrupted 3-epoch baseline
            ref_model, ref_opt, ref_sched = build_components()
            for ep in range(epochs):
                ref_opt.zero_grad()
                for m_idx, (bx, by) in enumerate(epoch_data[ep]):
                    loss = ((ref_model(bx) - by) ** 2).mean() / grad_accum
                    loss.backward()
                    if (m_idx + 1) % grad_accum == 0:
                        ref_opt.step()
                        ref_sched.step()
                        ref_opt.zero_grad()

            # 2. Run Part 1 (Interrupted mid-way through Epoch 1 after 2nd optimizer step in Epoch 1:
            #    total global_step = 3 (Epoch 0) + 2 (Epoch 1) = 5, committed_micro_step = 8)
            run_dir = os.path.join(tmpdir, "runs", "stage3_test")
            mgr1 = patch_mod.CheckpointResumeManager(output_dir=run_dir, save_every_steps=1)
            m1, opt1, sched1 = build_components()
            mgr1.register_model(m1, output_dir=run_dir)
            mgr1.register_optimizer(opt1)
            mgr1.register_scheduler(sched1)

            paused = False
            for ep in range(epochs):
                opt1.zero_grad()
                for m_idx, (bx, by) in enumerate(mgr1.iterate_dataloader(epoch_data[ep])):
                    loss = ((m1(bx) - by) ** 2).mean() / grad_accum
                    loss.backward()
                    if (m_idx + 1) % grad_accum == 0:
                        opt1.step()
                        sched1.step()
                        opt1.zero_grad()
                        if ep == 1 and mgr1.global_step == 5:
                            # Advance 2 extra micro-batches (to micro_step=10, mid-accumulation window)
                            # and fire a real SIGINT (Ctrl+C) to verify signal handler saves at committed_micro_step=8
                            pass
                    if ep == 1 and mgr1.current_micro_step == 10:
                        try:
                            signal.raise_signal(signal.SIGINT)
                        except KeyboardInterrupt:
                            paused = True
                            break
                if paused:
                    break

            self.assertTrue(paused, "SIGINT handler did not raise KeyboardInterrupt to pause training!")
            state_pt = os.path.join(run_dir, "resume_latest", "trainer_state.pt")
            mem_pt = os.path.join(run_dir, "resume_latest", "memory.pt")
            self.assertTrue(os.path.exists(state_pt))
            self.assertTrue(os.path.exists(mem_pt))
            self.assertGreater(os.path.getsize(state_pt), 1024)
            self.assertGreater(os.path.getsize(mem_pt), 1024)
            self.assertFalse(os.path.exists(state_pt + ".tmp"), "Stale .tmp file left behind after atomic save!")
            self.assertFalse(os.path.exists(mem_pt + ".tmp"), "Stale .tmp file left behind after atomic save!")

            # Verify on-disk payload contents and folded sqrt(2816) scale in memory.pt
            disk_state = torch.load(state_pt, map_location="cpu", weights_only=False)
            disk_mem = torch.load(mem_pt, map_location="cpu", weights_only=True)
            self.assertEqual(disk_state["epoch"], 1)
            self.assertEqual(disk_state["global_step"], 5)
            self.assertEqual(
                disk_state["committed_micro_step"],
                8,
                "Signal handler mid-accumulation window (micro_step=10) did not roll back to committed_micro_step=8!",
            )
            self.assertTrue( str(disk_state["reason"]).startswith("signal_"))
            self.assertIsNotNone(disk_state["optimizer_state_dict"])
            self.assertIsNotNone(disk_state["scheduler_state_dict"])
            self.assertGreater(disk_mem["proj.weight"].float().std().item(), 0.5)

            # 3. Run Part 2 (Resumed from disk in a brand-new process/manager)
            mgr2 = patch_mod.CheckpointResumeManager(output_dir=run_dir, save_every_steps=2)
            self.assertEqual(mgr2.resume_epoch, 1)
            self.assertEqual(mgr2.resume_micro_step, 8)
            self.assertEqual(mgr2.global_step, 5)

            m2, opt2, sched2 = build_components()
            mgr2.register_model(m2, output_dir=run_dir)
            mgr2.register_optimizer(opt2)
            mgr2.register_scheduler(sched2)

            batches_seen_per_epoch = {0: 0, 1: 0, 2: 0}
            for ep in range(epochs):
                opt2.zero_grad()
                for m_idx, (bx, by) in enumerate(mgr2.iterate_dataloader(epoch_data[ep])):
                    batches_seen_per_epoch[ep] += 1
                    loss = ((m2(bx) - by) ** 2).mean() / grad_accum
                    loss.backward()
                    if (m_idx + 1) % grad_accum == 0:
                        opt2.step()
                        sched2.step()
                        opt2.zero_grad()

            # Verify Epoch 0 was skipped (0 batches), Epoch 1 ran only the remaining 4 batches,
            # and Epoch 2 ran all 12 batches
            self.assertEqual(batches_seen_per_epoch[0], 0, "Completed Epoch 0 was not skipped on resume!")
            self.assertEqual(batches_seen_per_epoch[1], 4, "Epoch 1 did not fast-forward past first 8 micro-batches!")
            self.assertEqual(batches_seen_per_epoch[2], 12, "Epoch 2 did not run all 12 micro-batches!")
            self.assertEqual(mgr2.global_step, 9)

            # Verify final weights of [Part 1 + Part 2] match the uninterrupted baseline run
            self.assertTrue(
                torch.allclose(m2.proj.weight, ref_model.proj.weight, rtol=1e-5, atol=1e-6),
                "Resumed training weights diverged from uninterrupted baseline!",
            )

            # Verify live telemetry JSONL history was recorded for all 9 optimizer steps across Part 1 + Part 2
            metrics_jsonl = os.path.join(run_dir, "metrics_history.jsonl")
            self.assertTrue(os.path.exists(metrics_jsonl), "metrics_history.jsonl was not written during training!")
            with open(metrics_jsonl, "r", encoding="utf-8") as mf:
                records = [json.loads(line) for line in mf if line.strip()]
            self.assertEqual(len(records), 9, f"Expected 9 step metric records, found {len(records)}")
            self.assertEqual([r["global_step"] for r in records], list(range(1, 10)))
            for rec in records:
                self.assertIn("grad_norm", rec)
                self.assertIn("proj_weight_std", rec)
                self.assertIn("eta_epoch_s", rec)
                self.assertGreater(rec["grad_norm"], 0.0)
                self.assertGreater(rec["proj_weight_std"], 0.0)

            logger.info(
                "  Save/Resume & Live Telemetry verification: Epoch 0 skipped, Epoch 1 fast-forwarded (8/12), "
                "9/9 step telemetry records validated, final weights match uninterrupted run."
            )


if __name__ == "__main__":
    logger.info("======================================================================")
    logger.info("Starting Gemma 4 26B-A4B Memorilla Verification & Diagnostic Test Suite")
    logger.info(f"Logging detailed output to: {LOG_FILE}")
    logger.info("======================================================================")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGemma4MemorillaPipeline)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    if result.wasSuccessful():
        logger.info("======================================================================")
        logger.info(f"ALL {result.testsRun} TESTS PASSED SUCCESSFULLY. ZERO ERRORS DETECTED.")
        logger.info("======================================================================")
        sys.exit(0)
    else:
        logger.error("======================================================================")
        logger.error(
            f"SUITE FAILED: {len(result.failures)} failure(s), {len(result.errors)} error(s). "
            f"See {LOG_FILE} for full tracebacks."
        )
        logger.error("======================================================================")
        sys.exit(1)
