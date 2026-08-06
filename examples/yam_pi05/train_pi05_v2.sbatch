#!/bin/bash
#SBATCH -A kempner_ydu_lab
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/logs/%x-%j.out
# Parameterized pi05 fine-tune (ultrawide dataset era). Submit with -J <tag> and env:
#   RUN_TAG      run/wandb name, e.g. pi05-uw-top-12k (default: $SLURM_JOB_NAME)
#   DATA_ROOT_V2 dataset root (fixedstats overlay)
#   RENAME_MAP_V2  dataset-cam -> pi05-slot JSON. Omitting a slot (e.g. no top ->
#                  no base_0_rgb mapping) triggers pi05's NATIVE missing-camera
#                  masking (-1 pad + zero attn mask, modeling_pi05.py:1009-1014),
#                  in-distribution with openpi pretraining — better than black frames.
#   MAIN_OUT_OVERRIDE  adopt an existing run dir (resubmission resume)
# Example:
#   sbatch -p kempner_rtx -N 1 -t 12:00:00 --gres=gpu:8 -J pi05-uw-top --export=ALL train_pi05_v2.sbatch
#
# Carries every fix from the realsense pi05 run (job 37421128/37452745):
#   fp32 (bf16 AdamW-rounding trap), wandb staging to /tmp + disable_artifact
#   (netscratch EDQUOT crash), POLICY_SRC dropped on resume (--policy.path
#   preempts resume resolution), weights-loaded gate (silent random-init),
#   fixedstats overlay (#4156 quantile aggregation bug).

set -euo pipefail
source /n/netscratch/ydu_lab/Lab/asethi/molmoact-ft/env.sh   # tokens + wandb creds
export HL=/n/holylabs/kempner_ydu_lab/Lab/asethi
export HF_HOME=$HL/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_DATA_DIR=/tmp/wandb-data-$USER
export WANDB_ARTIFACT_DIR=/tmp/wandb-artifacts-$USER

export VENV=/n/netscratch/ydu_lab/Lab/asethi/venvs/lerobot-aq
export RUN_TAG=${RUN_TAG:-$SLURM_JOB_NAME}
export DATA_ROOT=${DATA_ROOT_V2:?set DATA_ROOT_V2}
export RENAME_MAP=${RENAME_MAP_V2:?set RENAME_MAP_V2}
export NNODES=$SLURM_JOB_NUM_NODES
export WORLD=$((NNODES * ${GPUS_PER_NODE:-8}))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
export TFS_JSON=$(cat $HL/molmoact-ft/tfs_run2.json)
LOGFILE=$HL/molmoact-ft/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out

TRAIN_STEPS=12000
export TRAIN_STEPS
echo "tag=$RUN_TAG data=$DATA_ROOT world=$WORLD steps=$TRAIN_STEPS branch=$(git -C ~/lerobot-aq rev-parse --abbrev-ref HEAD)@$(git -C ~/lerobot-aq rev-parse --short HEAD)"
echo "rename_map=$RENAME_MAP"

run_train() {
  # env args: RUN_STEPS RUN_WARMUP RUN_SAVE RUN_LOGF RUN_OUT RUN_NAME RUN_WANDB
  if [ -e "$RUN_OUT/checkpoints/last" ]; then
    echo "resuming from $RUN_OUT/checkpoints/last"
    export RUN_RESUME="--resume=true --config_path=$RUN_OUT/checkpoints/last/pretrained_model/train_config.json"
    # NO --policy.path on resume: its branch preempts resume resolution
    # (checkpoint_path=None crash) and would reload BASE weights.
    export POLICY_SRC=""
  else
    export RUN_RESUME=""
    export POLICY_SRC="--policy.path=lerobot/pi05_base"
  fi
  srun --kill-on-bad-exit=1 --ntasks=$NNODES --ntasks-per-node=1 bash -c '
  $VENV/bin/accelerate launch \
    --multi_gpu --num_machines=$NNODES --num_processes=$WORLD \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
    --same_network --mixed_precision=no \
    -m lerobot.scripts.lerobot_train \
    $RUN_RESUME \
    --dataset.repo_id=brandonyang/yam-ultrawide-teleop \
    --dataset.root=$DATA_ROOT \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs="$TFS_JSON" \
    --dataset.eval_split=0.0 \
    --rename_map="$RENAME_MAP" \
    $POLICY_SRC \
    --policy.device=cuda \
    --policy.dtype=float32 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=$RUN_WARMUP \
    --policy.scheduler_decay_steps=$RUN_STEPS \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.push_to_hub=false \
    --steps=$RUN_STEPS --batch_size=${BS:-8} --seed=1000 \
    --num_workers=8 --dataloader_multiprocessing_context=spawn \
    --tolerance_s=0.0001 \
    --log_freq=$RUN_LOGF --save_checkpoint=true --save_freq=$RUN_SAVE \
    --env_eval_freq=0 --eval_steps=0 \
    --wandb.disable_artifact=true \
    $RUN_WANDB \
    --job_name=$RUN_NAME --output_dir=$RUN_OUT
  '
}

check_weights_loaded() {
  if grep -q "Returning model without loading pretrained weights" "$LOGFILE" \
     || grep -q "Warning: Could not load state dict" "$LOGFILE"; then
    echo "FATAL: pretrained weight load failed (random init detected)"; exit 1
  fi
  grep -q "Loaded state dict from model.safetensors" "$LOGFILE" \
    || { echo "FATAL: no weight-load confirmation in log"; exit 1; }
  echo "WEIGHTS_LOADED_OK"
}

MAIN_OUT=${MAIN_OUT_OVERRIDE:-$HL/molmoact-ft/runs/${RUN_TAG}-${WORLD}gpu-$SLURM_JOB_ID}

# ---- Phase 1: 25-step canary (skipped once main run has checkpoints) ----
if [ ! -e "$MAIN_OUT/checkpoints" ]; then
  export RUN_STEPS=25 RUN_WARMUP=5 RUN_SAVE=25 RUN_LOGF=5 RUN_WANDB="--wandb.enable=false"
  export RUN_OUT=$HL/molmoact-ft/runs/canary-$RUN_TAG-$SLURM_JOB_ID RUN_NAME=canary-$RUN_TAG
  rm -rf "$RUN_OUT"
  echo "=== CANARY START $(date) ==="
  run_train
  check_weights_loaded
  echo "=== CANARY_OK $(date) ==="
fi

# ---- Phase 2: real run ----
export RUN_STEPS=$TRAIN_STEPS
export RUN_WARMUP=600
export RUN_SAVE=1000
export RUN_LOGF=20
export RUN_WANDB="--wandb.enable=true --wandb.entity=aq-robotics --wandb.project=molmoact2-yam-oranges"
export RUN_OUT=$MAIN_OUT
export RUN_NAME=$RUN_TAG
echo "=== TRAIN START $(date) ==="
run_train
check_weights_loaded
echo "=== TRAIN_OK $(date) ==="
