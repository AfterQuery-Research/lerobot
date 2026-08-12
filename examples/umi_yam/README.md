# Four all-182 long-gripper UMI/YAM training runs

`train_four_models.sbatch` is the only launcher. Its four profiles share the exact task
`pick up oranges and place them in the bowl`, horizon 24, seed 1000, 12,000 optimizer steps,
global batch 64, and checkpoints every 1,000 steps. The selected topology sets the per-rank batch.

| Profile | Action | Dataset tail |
| --- | --- | --- |
| `molmoact2-ee20` | independent-arm query-anchored EE, 20-D | already removed by the compact EE sidecar; launcher requires `drop_n_last_frames=0` |
| `pi05-ee20` | same 20-D EE contract | same |
| `molmoact2-joint14` | published I2RT `7ed46f4` 220 mm-grasp joint positions, 14-D | launcher requires `drop_n_last_frames=24` |
| `pi05-joint14` | same published 14-D joint contract | same |

Supported allocations preserve the same global batch and optimizer-step contract:

| `UMI_YAM_TOPOLOGY` | Allocation | Batch/rank | Partition |
| --- | --- | --- | --- |
| `2x8` (default) | 2 nodes × 8 GPUs | 4 | `kempner_requeue` |
| `4x4` | 4 nodes × 4 GPUs | 4 | `kempner_requeue` |
| `2x4` | 2 nodes × 4 GPUs | 8 | `kempner_requeue` |
| `protected1x4` | 1 node × 4 GPUs | 16 | `kempner_rtx` |

The Slurm request and `UMI_YAM_TOPOLOGY` must agree; the launcher verifies both before touching model
state. Keep the same topology when resuming because checkpoint validation pins world size and per-rank
batch. Use `protected1x4` only when the protected four-GPU allocation has been explicitly coordinated.

Model pins are MolmoAct2 `8dcbed66f2380e4393189c303ea72488eb9e63c2`, FAST tokenizer
`d45593b4c863d0bc1ca064f8b352fa16b75c38e8`, and Pi0.5
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` with PaliGemma tokenizer
`35e4f46485b4d07967e7e9935bc3786aad50687c`. Dataset pins are Cartesian
`a29ae6a5531584fb950c7bb3bb5895f18421b108` and published joints
`387d696eb36de411a11c8b2612a6f19822ca3a53`. The 182 episodes and 179,951 frames yield 175,583 H24
queries after tail removal; no episode is excluded.

## Prepare artifacts once

Run once on a suitable compute allocation using the clean training checkout. The EE builder combines
the pinned Cartesian poses with row-aligned joint-snapshot grippers without duplicate joint videos.

```bash
export UMI_YAM_CODE_ROOT=/path/to/clean/lerobot
export UMI_YAM_CODE_REVISION=$(git -C "$UMI_YAM_CODE_ROOT" rev-parse HEAD)
export UMI_YAM_VENV=/path/to/training-venv
export UMI_YAM_HF_HOME=/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache
export UMI_YAM_EE_CACHE_ROOT=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-long-gripper-ee20-h24-cache
export UMI_YAM_JOINT_ROOT=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable

test "$(git -C "$UMI_YAM_CODE_ROOT" rev-parse HEAD)" = "$UMI_YAM_CODE_REVISION"
test -z "$(git -C "$UMI_YAM_CODE_ROOT" status --porcelain)"
test ! -e "$UMI_YAM_EE_CACHE_ROOT"
PYTHONNOUSERSITE=1 PYTHONPATH="$UMI_YAM_CODE_ROOT/src" HF_HOME="$UMI_YAM_HF_HOME" \
  "$UMI_YAM_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.umi_yam_ee_dataset import (
    JOINT_REPO_ID,
    JOINT_REVISION,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    materialize_cache,
)

root = Path(os.environ["UMI_YAM_EE_CACHE_ROOT"])
if root.exists():
    raise FileExistsError(root)
source = LeRobotDataset(
    SOURCE_REPO_ID,
    revision=SOURCE_REVISION,
    video_backend="pyav",
    return_uint8=True,
)
joint = LeRobotDataset(
    JOINT_REPO_ID,
    revision=JOINT_REVISION,
    delta_timestamps=None,
    download_videos=False,
)
materialize_cache(source, joint, root)
PY

test ! -e "$UMI_YAM_JOINT_ROOT"
PYTHONNOUSERSITE=1 PYTHONPATH="$UMI_YAM_CODE_ROOT/src" HF_HOME="$UMI_YAM_HF_HOME" \
  "$UMI_YAM_VENV/bin/python" -m lerobot.scripts.materialize_umi_yam_joint_dataset \
  --output-root "$UMI_YAM_JOINT_ROOT" \
  --repo-id ASethi04/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable
```

Both builders write `meta/payload.sha256`; the launcher refuses missing, changed, symlinked, or
un-pinned payloads. Do not let a training job create either artifact concurrently.

## Configure once

W&B defaults to offline under `UMI_YAM_RUN_ROOT`; online mode requires `UMI_YAM_WANDB_MODE=online`
and `WANDB_API_KEY` in a mode-600 environment file.

```bash
export UMI_YAM_CODE_ROOT=/path/to/clean/lerobot
export UMI_YAM_CODE_REVISION=$(git -C "$UMI_YAM_CODE_ROOT" rev-parse HEAD)
export UMI_YAM_VENV=/path/to/training-venv
export UMI_YAM_RUN_ROOT=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-runs
export UMI_YAM_RUN_ID=long-gripper182-i2rt7ed-v1
export UMI_YAM_HF_HOME=/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache
export UMI_YAM_PI05_TOKENIZER_HUB_CACHE=/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache/hub
export UMI_YAM_EE_CACHE_ROOT=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-long-gripper-ee20-h24-cache
export UMI_YAM_JOINT_ROOT=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable
mkdir -p /n/netscratch/ydu_lab/Lab/asethi/umi-yam-runs/slurm
# Optional for online W&B only:
# export UMI_YAM_WANDB_MODE=online
# export UMI_YAM_ENV_FILE=/path/to/mode-600.env
```

The launcher rejects an EE source-root override. Each command below pins the selected payload
manifest; every listed file is rehashed before either node supervisor starts.

## Validate, then launch

Parse the request without submitting it:

```bash
sbatch --test-only examples/umi_yam/train_four_models.sbatch
```

Run one allocated dry-run per data kind. Each checks one semantic sample, provenance, QOS `normal`,
the selected topology, and every visible typed GPU, then exits before model loading.

```bash
export UMI_YAM_PROFILE=molmoact2-ee20
export UMI_YAM_PAYLOAD_MANIFEST_SHA256=$(sha256sum "$UMI_YAM_EE_CACHE_ROOT/meta/payload.sha256" | awk '{print $1}')
sbatch --export=ALL,UMI_YAM_DRY_RUN=1 examples/umi_yam/train_four_models.sbatch

export UMI_YAM_PROFILE=molmoact2-joint14
export UMI_YAM_PAYLOAD_MANIFEST_SHA256=$(sha256sum "$UMI_YAM_JOINT_ROOT/meta/payload.sha256" | awk '{print $1}')
sbatch --export=ALL,UMI_YAM_DRY_RUN=1 examples/umi_yam/train_four_models.sbatch
```

After both dry-runs print `ALLOCATION_OK`, `DATASET_OK`, `PROFILE_OK`, and `DRY_RUN_OK`, submit the
four exact profiles with the corresponding manifest hash. The commands below use the default `2x8`;
for fragmented capacity, override the request and topology together, for example:

```bash
# 4 nodes × 4 GPUs, still world size 16 and batch 64.
sbatch --nodes=4 --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4 \
  --cpus-per-task=64 --export=ALL,UMI_YAM_TOPOLOGY=4x4 \
  examples/umi_yam/train_four_models.sbatch

# 2 nodes × 4 GPUs, world size 8 with batch 8/rank.
sbatch --nodes=2 --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4 \
  --cpus-per-task=64 --export=ALL,UMI_YAM_TOPOLOGY=2x4 \
  examples/umi_yam/train_four_models.sbatch

# Coordinated protected allocation. Pi0.5 batch 16/rank is validated on RTX Pro 6000.
sbatch --partition=kempner_rtx --nodes=1 \
  --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4 --cpus-per-task=48 \
  --time=24:00:00 --export=ALL,UMI_YAM_TOPOLOGY=protected1x4 \
  examples/umi_yam/train_four_models.sbatch
```

Default `2x8` submissions:

```bash
unset UMI_YAM_DRY_RUN UMI_YAM_RESUME_CONFIG
export UMI_YAM_PAYLOAD_MANIFEST_SHA256=$(sha256sum "$UMI_YAM_EE_CACHE_ROOT/meta/payload.sha256" | awk '{print $1}')
for UMI_YAM_PROFILE in molmoact2-ee20 pi05-ee20; do
  export UMI_YAM_PROFILE
  sbatch --export=ALL examples/umi_yam/train_four_models.sbatch
done

export UMI_YAM_PAYLOAD_MANIFEST_SHA256=$(sha256sum "$UMI_YAM_JOINT_ROOT/meta/payload.sha256" | awk '{print $1}')
for UMI_YAM_PROFILE in molmoact2-joint14 pi05-joint14; do
  export UMI_YAM_PROFILE
  sbatch --export=ALL examples/umi_yam/train_four_models.sbatch
done
```

Fresh launches refuse an existing output. Manual resume verifies the numeric checkpoint, training state,
world/batch size, profile, dataset, dimensions, and save boundary. Requeue selects the highest complete
checkpoint even if `last` is stale, ignores incomplete higher writes, and fails on malformed state. Before
the first checkpoint it preserves partial output as `.precheckpoint-rN` and restarts at step 0.

```bash
export UMI_YAM_RESUME_CONFIG="$UMI_YAM_RUN_ROOT/$UMI_YAM_RUN_ID/$UMI_YAM_PROFILE/checkpoints/004000/pretrained_model/train_config.json"
sbatch --export=ALL examples/umi_yam/train_four_models.sbatch
```

Never point two jobs at the same profile output directory. Except for the explicitly coordinated
`protected1x4` mode, use `kempner_requeue`. Do not use `kempner`, `kempner_h100`, or `kempner_h200`;
all modes require the exact typed RTX Pro 6000 request.
