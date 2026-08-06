"""Login-node preflight for the pi05 YAM fine-tune (light: no 14.5GB weight load).

The full preflight gets OOM-killed by login-node memory caps during model
loading, so weight-load + forward validation is delegated to the sbatch's
25-step GPU canary (gated by check_weights_loaded). This script validates
everything else:
  1. draccus parse + validate() of the EXACT training argv (--policy.path merge)
  2. dataset load (transforms JSON parse included) + sample decode
  3. q01/q99 stats vs directly-computed global quantiles (upstream bug #4156)
  4. processor pipelines with lerobot_train's overrides + one real batch through
     the PREPROCESSOR (rename -> quantile normalize -> state discretize ->
     tokenize offline) -- everything but the model itself

Run: VENV/bin/python -u preflight_pi05_light.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = "/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HL = "/n/holylabs/kempner_ydu_lab/Lab/asethi"
DATA_ROOT = f"{HL}/molmoact-ft/data/yam-vive-teleop-fixedstats"
TFS_JSON = Path(f"{HL}/molmoact-ft/tfs_run2.json").read_text().strip()
RENAME_MAP = (
    '{"observation.images.top":"observation.images.base_0_rgb",'
    '"observation.images.left":"observation.images.left_wrist_0_rgb",'
    '"observation.images.right":"observation.images.right_wrist_0_rgb"}'
)
STEPS = 12000

ARGV = [
    "--dataset.repo_id=brandonyang/yam-vive-teleop",
    f"--dataset.root={DATA_ROOT}",
    "--dataset.video_backend=pyav",
    "--dataset.image_transforms.enable=true",
    f"--dataset.image_transforms.tfs={TFS_JSON}",
    "--dataset.eval_split=0.0",
    f"--rename_map={RENAME_MAP}",
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
    f"--steps={STEPS}",
    "--batch_size=8",
    "--seed=1000",
    "--num_workers=8",
    "--dataloader_multiprocessing_context=spawn",
    "--tolerance_s=0.0001",
    "--log_freq=20",
    "--save_checkpoint=true",
    "--save_freq=1000",
    "--env_eval_freq=0",
    "--eval_steps=0",
    "--wandb.enable=false",
    "--job_name=pi05-preflight",
    f"--output_dir={HL}/molmoact-ft/runs/pi05-preflight-parse-only",
]

print("=== 1. draccus parse + validate of exact training argv ===", flush=True)
import draccus
import torch

torch.set_num_threads(4)

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401  registers 'pi05'

# Mirror parser.wrap(): strip --policy.* from the draccus parse, keep full argv
# visible so validate()'s get_path_arg/get_cli_overrides recover path+overrides.
sys.argv = ["lerobot-train"] + ARGV
draccus_args = [a for a in ARGV if not a.startswith("--policy.")]
cfg = draccus.parse(config_class=TrainPipelineConfig, args=draccus_args)
cfg.validate()
p = cfg.policy
print("parse+validate OK | type", p.type, "| pretrained_path", p.pretrained_path)
print("  dtype", p.dtype, "| chunk", p.chunk_size, "| n_action_steps", p.n_action_steps,
      "| grad_ckpt", p.gradient_checkpointing, "| compile", p.compile_model,
      "| push_to_hub", p.push_to_hub)
print("  optimizer:", type(cfg.optimizer).__name__, "lr", cfg.optimizer.lr,
      "betas", cfg.optimizer.betas, "wd", cfg.optimizer.weight_decay,
      "clip", cfg.optimizer.grad_clip_norm)
print("  scheduler:", type(cfg.scheduler).__name__, "peak", cfg.scheduler.peak_lr,
      "decay_lr", cfg.scheduler.decay_lr, "warmup", cfg.scheduler.num_warmup_steps,
      "decay", cfg.scheduler.num_decay_steps)
print("  normalization:", {k: v.value for k, v in p.normalization_mapping.items()})
print("  input_features:", sorted(p.input_features.keys()))
print("  rename_map:", cfg.rename_map, flush=True)
assert p.dtype == "float32", "dtype must be float32 (bf16 AdamW trap)"
assert p.chunk_size == 30 and p.n_action_steps == 30
assert not p.compile_model, "compile_model must stay off (upstream #4178)"
assert not p.push_to_hub
assert cfg.scheduler.num_warmup_steps == 600 and cfg.scheduler.num_decay_steps == STEPS
assert len(cfg.rename_map) == 3

print("=== 2. dataset load (with image transforms) + sample decode ===", flush=True)
from lerobot.datasets.factory import make_train_eval_datasets

dataset, eval_dataset = make_train_eval_datasets(cfg)
print("episodes", dataset.meta.total_episodes, "| frames", dataset.meta.total_frames,
      "| fps", dataset.meta.fps, "| eval_dataset", eval_dataset)
assert dataset.meta.total_episodes == 80 and dataset.meta.total_frames == 74927
item = dataset[0]
for k in ("observation.images.top", "observation.images.left", "observation.images.right",
          "observation.state", "action"):
    print(f"  {k}: {tuple(item[k].shape)} {item[k].dtype}")
print("  task:", repr(item.get("task")), flush=True)
assert item["action"].shape == (30, 14), "delta_timestamps must give a 30-step chunk"

print("=== 3. quantile stats vs direct global computation (#4156) ===", flush=True)
import numpy as np
import pyarrow.parquet as pq

stats = dataset.meta.stats
parquets = sorted(Path(DATA_ROOT).glob("data/**/*.parquet"))
print("  parquet files:", len(parquets))
cols = {"action": [], "observation.state": []}
for f in parquets:
    t = pq.read_table(f, columns=list(cols.keys()))
    for c in cols:
        cols[c].append(np.stack(t.column(c).to_numpy(zero_copy_only=False)))
worst = 0.0
for c, chunks in cols.items():
    arr = np.concatenate(chunks, axis=0)
    assert arr.shape == (74927, 14), f"{c}: unexpected shape {arr.shape}"
    for qname, qv in (("q01", 0.01), ("q99", 0.99)):
        direct = np.quantile(arr, qv, axis=0)
        stored = np.asarray(stats[c][qname], dtype=np.float64).reshape(-1)
        rng = np.asarray(stats[c]["q99"], dtype=np.float64).reshape(-1) - np.asarray(
            stats[c]["q01"], dtype=np.float64).reshape(-1)
        rel = np.abs(direct - stored) / np.maximum(np.abs(rng), 1e-6)
        worst = max(worst, float(rel.max()))
        print(f"  {c}.{qname}: max|direct-stored| = {np.abs(direct - stored).max():.5f} "
              f"(max {100 * rel.max():.2f}% of q-range)")
print(f"  WORST deviation: {100 * worst:.2f}% of quantile range", flush=True)
if worst > 0.05:
    print("  QUANTILE_STATS_SUSPECT — stored stats deviate >5% from direct global quantiles")
else:
    print("  quantile stats OK (within 5% of direct computation)")

print("=== 4. processor pipelines + one real batch through the preprocessor ===", flush=True)
from lerobot.configs import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

# Mirror make_policy's feature fixup (factory.py:290-306): output_features come
# from the dataset (action -> [14]); non-empty checkpoint input_features stay.
features = dataset_to_policy_features(dataset.meta.features)
p.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
if not p.input_features:
    p.input_features = {k: ft for k, ft in features.items() if k not in p.output_features}
print("  output_features:", {k: ft.shape for k, ft in p.output_features.items()})

preprocessor_overrides = {
    "device_processor": {"device": "cpu"},
    "normalizer_processor": {
        "features": {**p.input_features, **p.output_features},
        "norm_map": p.normalization_mapping,
        "stats": dataset.meta.stats,
    },
    "rename_observations_processor": {"rename_map": cfg.rename_map},
}
postprocessor_overrides = {
    "unnormalizer_processor": {
        "features": p.output_features,
        "norm_map": p.normalization_mapping,
        "stats": dataset.meta.stats,
    },
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=p,
    pretrained_path=p.pretrained_path,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides=preprocessor_overrides,
    postprocessor_overrides=postprocessor_overrides,
)
print("  pipelines OK:", type(preprocessor).__name__, type(postprocessor).__name__, flush=True)

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
proc = preprocessor(batch)
img_keys = sorted(k for k in proc if str(k).startswith("observation.images."))
print("  post-preprocess image keys:", img_keys)
assert "observation.images.base_0_rgb" in proc, "rename_map did not apply"
assert "observation.images.top" not in proc, "old camera key survived rename"
state = proc["observation.state"]
print(f"  normalized state: shape {tuple(state.shape)} range [{float(state.min()):.3f}, {float(state.max()):.3f}]")
act = proc["action"]
print(f"  normalized action: shape {tuple(act.shape)} range [{float(act.min()):.3f}, {float(act.max()):.3f}]")
toks = proc.get("input_ids", None)
if toks is None:
    tok_keys = [k for k in proc if "token" in str(k).lower() or "input_ids" in str(k)]
    print("  tokenizer outputs:", tok_keys)
else:
    print(f"  input_ids: shape {tuple(toks.shape)}")
assert act.shape[-1] == 14 and act.shape[-2] == 30

print()
print("PREFLIGHT_LIGHT_OK", flush=True)
