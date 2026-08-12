#!/usr/bin/env bash
#SBATCH --job-name=umi-yam-four
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/asethi/umi-yam-runs/slurm/%x-%j.out

set -euo pipefail

if [[ -n ${UMI_YAM_ENV_FILE:-} ]]; then
  [[ -r $UMI_YAM_ENV_FILE ]] || { echo "UMI_YAM_ENV_FILE is not readable" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$UMI_YAM_ENV_FILE"
fi

: "${UMI_YAM_PROFILE:?select molmoact2-ee20, pi05-ee20, molmoact2-joint14, or pi05-joint14}"
: "${UMI_YAM_CODE_ROOT:?point to the clean training checkout}"
: "${UMI_YAM_CODE_REVISION:?pin the exact git commit in UMI_YAM_CODE_ROOT}"
: "${UMI_YAM_VENV:?point to the training virtual environment}"
: "${UMI_YAM_RUN_ROOT:?set the parent directory for all four runs}"
: "${UMI_YAM_RUN_ID:?set one stable identifier shared by the four runs}"
: "${UMI_YAM_PAYLOAD_MANIFEST_SHA256:?pin the selected meta/payload.sha256 file}"

readonly TASK="pick up oranges and place them in the bowl"
readonly MOLMO_REPO="allenai/MolmoAct2-BimanualYAM"
readonly MOLMO_REVISION="8dcbed66f2380e4393189c303ea72488eb9e63c2"
readonly MOLMO_TOKENIZER_REPO="allenai/MolmoAct2-FAST-Tokenizer"
readonly MOLMO_TOKENIZER_REVISION="d45593b4c863d0bc1ca064f8b352fa16b75c38e8"
readonly PI05_REPO="lerobot/pi05_base"
readonly PI05_REVISION="b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
readonly EE_REPO="brandonyang/dual-lidar-combined-filtered-long-gripper"
readonly EE_REVISION="a29ae6a5531584fb950c7bb3bb5895f18421b108"
readonly JOINT_REPO="ASethi04/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable"
readonly DEFAULT_EE_CACHE_ROOT="/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-long-gripper-ee20-h24-cache"
readonly DEFAULT_JOINT_ROOT="/n/netscratch/ydu_lab/Lab/asethi/umi-yam-data/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable"
readonly WORLD_SIZE=16
readonly GPUS_PER_NODE=8
readonly PER_RANK_BATCH=4
readonly ACTION_HORIZON=24

case "$UMI_YAM_PROFILE" in
  molmoact2-ee20)    MODEL=molmoact2; DATA_KIND=ee;    ACTION_DIM=20; DROP_LAST=0 ;;
  pi05-ee20)         MODEL=pi05;      DATA_KIND=ee;    ACTION_DIM=20; DROP_LAST=0 ;;
  molmoact2-joint14) MODEL=molmoact2; DATA_KIND=joint; ACTION_DIM=14; DROP_LAST=24 ;;
  pi05-joint14)      MODEL=pi05;      DATA_KIND=joint; ACTION_DIM=14; DROP_LAST=24 ;;
  *) echo "unknown UMI_YAM_PROFILE=$UMI_YAM_PROFILE" >&2; exit 2 ;;
esac
[[ $((WORLD_SIZE * PER_RANK_BATCH)) == 64 && $ACTION_HORIZON == 24 ]]
if [[ $DATA_KIND == ee ]]; then
  [[ $ACTION_DIM == 20 && $DROP_LAST == 0 ]]
  [[ -z ${UMI_YAM_EE_SOURCE_ROOT:-} ]] || {
    echo "UMI_YAM_EE_SOURCE_ROOT is forbidden: EE source must use the pinned Hub snapshot" >&2
    exit 2
  }
  EE_CACHE_ROOT=${UMI_YAM_EE_CACHE_ROOT:-$DEFAULT_EE_CACHE_ROOT}
  DATA_REPO=$EE_REPO
  DATA_ROOT=
  DATA_REVISION=$EE_REVISION
  DATA_MANIFEST=$EE_CACHE_ROOT/meta/umi_yam_ee_long_gripper.json
  PAYLOAD_ROOT=$EE_CACHE_ROOT
else
  [[ $ACTION_DIM == 14 && $DROP_LAST == 24 ]]
  DATA_REPO=$JOINT_REPO
  DATA_ROOT=${UMI_YAM_JOINT_ROOT:-$DEFAULT_JOINT_ROOT}
  DATA_REVISION=
  DATA_MANIFEST=$DATA_ROOT/meta/materialization.json
  PAYLOAD_ROOT=$DATA_ROOT
fi

[[ $UMI_YAM_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || {
  echo "UMI_YAM_RUN_ID contains unsupported characters" >&2
  exit 2
}
[[ $UMI_YAM_PAYLOAD_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "UMI_YAM_PAYLOAD_MANIFEST_SHA256 must be a lowercase SHA-256" >&2
  exit 2
}

CODE_ROOT=$(realpath "$UMI_YAM_CODE_ROOT")
VENV=$(realpath "$UMI_YAM_VENV")
RUN_ROOT=$(realpath -m "$UMI_YAM_RUN_ROOT")
RUN_DIR=$RUN_ROOT/$UMI_YAM_RUN_ID/$UMI_YAM_PROFILE
readonly CODE_ROOT VENV RUN_ROOT RUN_DIR MODEL DATA_KIND ACTION_DIM DROP_LAST
test -x "$VENV/bin/python"
test -x "$VENV/bin/accelerate"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$UMI_YAM_CODE_REVISION"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"

export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE
export PYTHONPATH=$CODE_ROOT/src
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME=${UMI_YAM_HF_HOME:-/n/holylabs/kempner_ydu_lab/Lab/asethi/hf-cache}

mapfile -t ALLOCATED_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${SLURM_JOB_NUM_NODES:-0} == 2 && ${SLURM_NTASKS:-0} == 2 ]]
[[ ${#ALLOCATED_HOSTS[@]} == 2 && ${ALLOCATED_HOSTS[0]} != "${ALLOCATED_HOSTS[1]}" ]]
JOB_DESCRIPTION=$(scontrol show job -d -o "$SLURM_JOB_ID")
grep -Eq '(^|[[:space:]])QOS=normal([[:space:]]|$)' <<< "$JOB_DESCRIPTION"
grep -Eq 'gres/gpu=16([,[:space:]]|$)' <<< "$JOB_DESCRIPTION"
grep -Fq 'TresPerNode=gres/gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8' \
  <<< "$JOB_DESCRIPTION"

mapfile -t GPU_REPORT < <(
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=96 --exact bash -c \
    'printf "%s\t%s\n" "$(hostname -s)" "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)"'
)
[[ ${#GPU_REPORT[@]} == 2 ]]
declare -A GPU_HOSTS=()
TOTAL_GPUS=0
for row in "${GPU_REPORT[@]}"; do
  read -r host count <<< "$row"
  [[ $count == 8 ]]
  GPU_HOSTS[$host]=1
  TOTAL_GPUS=$((TOTAL_GPUS + count))
done
[[ ${#GPU_HOSTS[@]} == 2 && $TOTAL_GPUS == 16 ]]
echo "ALLOCATION_OK qos=normal nodes=2 supervisors=2 gpus_per_node=8 total_gpus=16"

PAYLOAD_MANIFEST=$PAYLOAD_ROOT/meta/payload.sha256
[[ -f $PAYLOAD_MANIFEST && ! -L $PAYLOAD_MANIFEST ]]
test "$(sha256sum "$PAYLOAD_MANIFEST" | awk '{print $1}')" = \
  "$UMI_YAM_PAYLOAD_MANIFEST_SHA256"
(cd "$PAYLOAD_ROOT" && sha256sum --quiet --strict -c meta/payload.sha256)
echo "PAYLOAD_OK manifest=$PAYLOAD_MANIFEST sha256=$UMI_YAM_PAYLOAD_MANIFEST_SHA256"

export DATA_KIND DATA_REPO DATA_ROOT DATA_REVISION DATA_MANIFEST ACTION_DIM DROP_LAST TASK
export UMI_YAM_EE_CACHE_ROOT=${EE_CACHE_ROOT:-}
readonly PREFLIGHT=$CODE_ROOT/examples/umi_yam/training_preflight.py
[[ -f $PREFLIGHT && ! -L $PREFLIGHT ]]
"$VENV/bin/python" "$PREFLIGHT" dataset

RESUME_CONFIG=${UMI_YAM_RESUME_CONFIG:-}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
[[ $RESTART_COUNT =~ ^[0-9]+$ ]] || { echo "invalid SLURM_RESTART_COUNT" >&2; exit 2; }
export RUN_DIR RESTART_COUNT REQUESTED_CONFIG=$RESUME_CONFIG UMI_YAM_PROFILE UMI_YAM_RUN_ID DATA_REPO
SELECTION=$("$VENV/bin/python" "$PREFLIGHT" resume)
SOURCE_ARGS=()
case "$SELECTION" in
  FRESH)
    RESUME_CONFIG=
    if [[ $RESTART_COUNT != 0 && ( -e $RUN_DIR || -L $RUN_DIR ) ]]; then
      [[ -d $RUN_DIR && ! -L $RUN_DIR ]]
      PRESERVED=$RUN_DIR.precheckpoint-r$RESTART_COUNT
      [[ ! -e $PRESERVED && ! -L $PRESERVED ]]
      mv -T -- "$RUN_DIR" "$PRESERVED"
      echo "AUTO_FRESH_RESTART restart=$RESTART_COUNT preserved=$PRESERVED"
    else
      [[ ! -e $RUN_DIR && ! -L $RUN_DIR ]] || {
        echo "fresh launch refuses existing RUN_DIR=$RUN_DIR; provide exact UMI_YAM_RESUME_CONFIG" >&2
        exit 2
      }
    fi
    ;;
  $'RESUME\t'*)
    RESUME_CONFIG=${SELECTION#*$'\t'}
    echo "RESUME_OK restart=$RESTART_COUNT config=$RESUME_CONFIG"
    SOURCE_ARGS=(--resume=true --config_path="$RESUME_CONFIG")
    ;;
  $'COMPLETE\t'*)
    echo "TRAINING_ALREADY_COMPLETE checkpoint=${SELECTION#*$'\t'}"
    exit 0
    ;;
  *) echo "invalid checkpoint selection: $SELECTION" >&2; exit 2 ;;
esac

readonly COLOR_TRANSFORMS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}}}'
DATASET_ARGS=(
  --dataset.repo_id="$DATA_REPO"
  --dataset.video_backend=pyav
  --dataset.eval_split=0
  --dataset.drop_n_last_frames="$DROP_LAST"
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.tfs="$COLOR_TRANSFORMS"
)
[[ -z $DATA_ROOT ]] || DATASET_ARGS+=(--dataset.root="$DATA_ROOT")
[[ -z $DATA_REVISION ]] || DATASET_ARGS+=(--dataset.revision="$DATA_REVISION")
if [[ $DATA_KIND == ee ]]; then
  DATASET_ARGS+=(--dataset.umi_yam_ee=true --dataset.umi_yam_ee_cache_root="$UMI_YAM_EE_CACHE_ROOT")
fi

if [[ -z $RESUME_CONFIG ]]; then
  if [[ $MODEL == molmoact2 ]]; then
    MOLMO_TOKENIZER_SNAPSHOT=$(MOLMO_TOKENIZER_REPO="$MOLMO_TOKENIZER_REPO" \
      MOLMO_TOKENIZER_REVISION="$MOLMO_TOKENIZER_REVISION" "$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["MOLMO_TOKENIZER_REPO"]
revision = os.environ["MOLMO_TOKENIZER_REVISION"]
snapshot = Path(snapshot_download(repo, revision=revision)).resolve()
assert snapshot.name == revision and snapshot.parent.name == "snapshots"
assert snapshot.parent.parent.name == "models--allenai--MolmoAct2-FAST-Tokenizer"
required = ("processor_config.json", "tokenizer.json", "tokenizer_config.json")
assert all((snapshot / name).is_file() for name in required)
processor = json.loads((snapshot / "processor_config.json").read_text())
assert processor["processor_class"] == "UniversalActionProcessor"
assert processor["scale"] == 10 and processor["vocab_size"] == 2048 and processor["min_token"] == -55
print(snapshot)
PY
    )
    readonly MOLMO_TOKENIZER_SNAPSHOT
    CONTROL_MODE="absolute joint pose"
    [[ $DATA_KIND == ee ]] && CONTROL_MODE="current-relative dual end-effector trajectory"
    SOURCE_ARGS=(
      --policy.type=molmoact2
      --policy.checkpoint_path="$MOLMO_REPO"
      --policy.checkpoint_revision="$MOLMO_REVISION"
      --policy.device=cuda
      --policy.action_mode=both
      --policy.inference_action_mode=continuous
      --policy.discrete_action_tokenizer="$MOLMO_TOKENIZER_SNAPSHOT"
      --policy.chunk_size=24
      --policy.n_action_steps=24
      --policy.n_obs_steps=1
      --policy.num_flow_timesteps=8
      --policy.expected_max_action_dim=32
      --policy.mask_action_dim_padding=true
      --policy.normalize_gripper=false
      --policy.model_dtype=bfloat16
      --policy.train_mode_vlm=lora
      --policy.lora_rank=64
      --policy.lora_alpha=16
      --policy.lora_dropout=0.05
      --policy.llm_residual_dropout=0.1
      --policy.freeze_embedding=true
      --policy.gradient_checkpointing=true
      --policy.optimizer_lr=1e-5
      --policy.optimizer_vit_lr=5e-6
      --policy.optimizer_connector_lr=5e-6
      --policy.optimizer_action_expert_lr=1e-4
      --policy.scheduler_warmup_steps=600
      --policy.scheduler_decay_steps=12000
      --policy.scheduler_decay_lr=1e-6
      --policy.setup_type="bimanual yam robotic arms in molmoact2"
      --policy.control_mode="$CONTROL_MODE"
      '--policy.image_keys=["observation.images.umi1","observation.images.umi2"]'
      --policy.push_to_hub=false
    )
  else
    PI05_SNAPSHOT=$(PI05_REPO="$PI05_REPO" PI05_REVISION="$PI05_REVISION" "$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["PI05_REPO"]
revision = os.environ["PI05_REVISION"]
snapshot = Path(snapshot_download(repo, revision=revision)).resolve()
assert snapshot.name == revision and snapshot.parent.name == "snapshots"
assert snapshot.parent.parent.name == "models--lerobot--pi05_base"
config = json.loads((snapshot / "config.json").read_text())
assert config["type"] == "pi05"
assert config["max_action_dim"] == config["max_state_dim"] == 32
assert (snapshot / "model.safetensors").is_file()
print(snapshot)
PY
    )
    readonly PI05_SNAPSHOT
    SOURCE_ARGS=(
      --policy.path="$PI05_SNAPSHOT"
      --policy.device=cuda
      --policy.dtype=float32
      --policy.chunk_size=24
      --policy.n_action_steps=24
      --policy.num_inference_steps=10
      --policy.use_relative_actions=false
      --policy.gradient_checkpointing=true
      --policy.compile_model=false
      --policy.optimizer_lr=2.5e-5
      --policy.optimizer_weight_decay=0.01
      --policy.scheduler_warmup_steps=600
      --policy.scheduler_decay_steps=12000
      --policy.scheduler_decay_lr=2.5e-6
      --policy.push_to_hub=false
      '--rename_map={"observation.images.umi1":"observation.images.left_wrist_0_rgb","observation.images.umi2":"observation.images.right_wrist_0_rgb"}'
    )
  fi
fi

JOB_NAME=umi-yam-$UMI_YAM_PROFILE-$UMI_YAM_RUN_ID
WANDB_RUN_MODE=${UMI_YAM_WANDB_MODE:-offline}
[[ $WANDB_RUN_MODE == offline || $WANDB_RUN_MODE == online ]] || {
  echo "UMI_YAM_WANDB_MODE must be offline or online" >&2
  exit 2
}
if [[ $WANDB_RUN_MODE == online && -z ${WANDB_API_KEY:-} ]]; then
  echo "UMI_YAM_WANDB_MODE=online requires WANDB_API_KEY" >&2
  exit 2
fi
TRAIN_ARGS=(
  "${SOURCE_ARGS[@]}"
  "${DATASET_ARGS[@]}"
  --batch_size=4
  --seed=1000
  --num_workers=8
  --dataloader_multiprocessing_context=spawn
  --tolerance_s=0.0001
  --steps=12000
  --log_freq=20
  --save_checkpoint=true
  --save_freq=1000
  --env_eval_freq=0
  --eval_steps=0
  --max_eval_samples=0
  --wandb.enable=true
  --wandb.entity=aq-robotics
  --wandb.project=molmoact2-yam-oranges
  --wandb.disable_artifact=true
  --wandb.mode="$WANDB_RUN_MODE"
  --wandb.run_id="$JOB_NAME"
  --job_name="$JOB_NAME"
  --output_dir="$RUN_DIR"
)

MASTER_ADDR=${ALLOCATED_HOSTS[0]}
MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
MIXED_PRECISION=no
[[ $MODEL == molmoact2 ]] && MIXED_PRECISION=bf16
export MASTER_ADDR MASTER_PORT MIXED_PRECISION CODE_ROOT VENV
echo "PROFILE_OK profile=$UMI_YAM_PROFILE model=$MODEL action_dim=$ACTION_DIM horizon=24 drop_last=$DROP_LAST global_batch=64 task=$TASK"

if [[ ${UMI_YAM_DRY_RUN:-0} == 1 ]]; then
  printf 'DRY_RUN_OK command: accelerate launch --multi_gpu --num_machines=2 --num_processes=16 '
  printf '%q ' --machine_rank='${SLURM_PROCID}' --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" --same_network --mixed_precision="$MIXED_PRECISION" \
    --dynamo_backend=no -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RUN_ROOT/$UMI_YAM_RUN_ID" "$RUN_ROOT/wandb" "$HF_HOME"
export WANDB_DIR=$RUN_ROOT/wandb
export WANDB_MODE=$WANDB_RUN_MODE
export UMI_YAM_CODE_ROOT=$CODE_ROOT UMI_YAM_VENV=$VENV

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=96 --exact --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    [[ $(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l) == 8 ]]
    exec "$UMI_YAM_VENV/bin/accelerate" launch \
      --multi_gpu --num_machines=2 --num_processes=16 \
      --machine_rank="$SLURM_PROCID" \
      --main_process_ip="$MASTER_ADDR" --main_process_port="$MASTER_PORT" \
      --same_network --mixed_precision="$MIXED_PRECISION" --dynamo_backend=no \
      -m lerobot.scripts.lerobot_train "$@"
  ' umi-yam-node-supervisor "${TRAIN_ARGS[@]}"
