"""Login-node preflight for BOTH ultrawide pi05 variants (with-top / no-top).

Mirrors train_pi05_v2.sbatch's exact argv per variant. The no-top variant's
load-bearing check: the preprocessor must tolerate an unmapped raw top key and
an absent base_0_rgb slot (model-side masking is validated by the GPU canary).

Run: VENV/bin/python -u preflight_pi05_uw.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = "/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HL = "/n/holylabs/kempner_ydu_lab/Lab/asethi"
DATA_ROOT = f"{HL}/molmoact-ft/data/yam-ultrawide-teleop-fixedstats"
TFS_JSON = Path(f"{HL}/molmoact-ft/tfs_run2.json").read_text().strip()
STEPS = 12000

RENAME_TOP = (
    '{"observation.images.top":"observation.images.base_0_rgb",'
    '"observation.images.left":"observation.images.left_wrist_0_rgb",'
    '"observation.images.right":"observation.images.right_wrist_0_rgb"}'
)
RENAME_NOTOP = (
    '{"observation.images.left":"observation.images.left_wrist_0_rgb",'
    '"observation.images.right":"observation.images.right_wrist_0_rgb"}'
)

import draccus
import numpy as np
import torch

torch.set_num_threads(4)

from lerobot.configs import FeatureType
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.factory import make_train_eval_datasets
from torch.utils.data import DataLoader


def build_argv(rename_map, tag):
    return [
        "--dataset.repo_id=brandonyang/yam-ultrawide-teleop",
        f"--dataset.root={DATA_ROOT}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        f"--dataset.image_transforms.tfs={TFS_JSON}",
        "--dataset.eval_split=0.0",
        f"--rename_map={rename_map}",
        "--policy.path=lerobot/pi05_base",
        "--policy.device=cuda",
        "--policy.dtype=float32",
        "--policy.chunk_size=30",
        "--policy.n_action_steps=30",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        "--policy.optimizer_lr=2.5e-5",
        "--policy.scheduler_warmup_steps=600",
        f"--policy.scheduler_decay_steps={STEPS}",
        "--policy.scheduler_decay_lr=2.5e-6",
        "--policy.push_to_hub=false",
        f"--steps={STEPS}", "--batch_size=8", "--seed=1000",
        "--num_workers=8", "--dataloader_multiprocessing_context=spawn",
        "--tolerance_s=0.0001", "--log_freq=20",
        "--save_checkpoint=true", "--save_freq=1000",
        "--env_eval_freq=0", "--eval_steps=0",
        "--wandb.disable_artifact=true", "--wandb.enable=false",
        f"--job_name={tag}",
        f"--output_dir={HL}/molmoact-ft/runs/{tag}-parse-only",
    ]


def preflight(tag, rename_map, expect_base):
    print(f"===== VARIANT {tag} =====")
    argv = build_argv(rename_map, tag)
    sys.argv = ["lerobot-train"] + argv
    draccus_args = [a for a in argv if not a.startswith("--policy.")]
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=draccus_args)
    cfg.validate()
    p = cfg.policy
    print("parse OK | dtype", p.dtype, "| chunk", p.chunk_size,
          "| warmup", cfg.scheduler.num_warmup_steps, "| decay", cfg.scheduler.num_decay_steps,
          "| rename", cfg.rename_map)
    assert p.dtype == "float32" and p.chunk_size == 30 and not p.compile_model
    assert len(cfg.rename_map) == (3 if expect_base else 2)

    dataset, _ = make_train_eval_datasets(cfg)
    assert dataset.meta.total_episodes == 75 and dataset.meta.total_frames == 108726
    item = dataset[0]
    print("dataset OK | eps", dataset.meta.total_episodes, "| frames", dataset.meta.total_frames,
          "| top", tuple(item["observation.images.top"].shape),
          "| left", tuple(item["observation.images.left"].shape),
          "| task", repr(item.get("task")))
    assert item["action"].shape == (30, 14)
    assert item["observation.images.left"].shape[-2:] == (312, 416)

    # stats sanity: corrected quantiles loaded
    q01 = np.asarray(dataset.meta.stats["action"]["q01"]).reshape(-1)
    q99 = np.asarray(dataset.meta.stats["action"]["q99"]).reshape(-1)
    assert (q99 > q01).all()

    features = dataset_to_policy_features(dataset.meta.features)
    p.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    pre, post = make_pre_post_processors(
        policy_cfg=p,
        pretrained_path=p.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "features": {**p.input_features, **p.output_features},
                "norm_map": p.normalization_mapping,
                "stats": dataset.meta.stats,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": p.output_features,
                "norm_map": p.normalization_mapping,
                "stats": dataset.meta.stats,
            },
        },
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    proc = pre(batch)
    img_keys = sorted(k for k in proc if str(k).startswith("observation.images."))
    print("post-preprocess image keys:", img_keys)
    assert "observation.images.left_wrist_0_rgb" in proc
    assert "observation.images.right_wrist_0_rgb" in proc
    if expect_base:
        assert "observation.images.base_0_rgb" in proc
        assert "observation.images.top" not in proc
    else:
        assert "observation.images.base_0_rgb" not in proc, "no-top variant must leave base slot empty"
        assert "observation.images.top" in proc, "raw top key should pass through untouched (ignored by model)"
    st = proc["observation.state"]
    act = proc["action"]
    print(f"normalized state [{float(st.min()):.3f},{float(st.max()):.3f}] "
          f"action [{float(act.min()):.3f},{float(act.max()):.3f}]")
    assert act.shape[-2:] == (30, 14)
    print(f"VARIANT_{tag}_OK\n")


preflight("pi05-uw-top", RENAME_TOP, expect_base=True)
preflight("pi05-uw-notop", RENAME_NOTOP, expect_base=False)
print("PREFLIGHT_UW_ALL_OK")
