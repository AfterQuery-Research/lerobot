#!/bin/bash
#SBATCH -J pi05-yam
#SBATCH -A kempner_ydu_lab
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/logs/pi05-%j.out
# Submit: sbatch -p kempner_rtx -N 1 -t 24:00:00 --gres=gpu:8 train_pi05.sbatch
#
# pi05 (pi-0.5) FULL fine-tune from lerobot/pi05_base on yam-vive-teleop.
# policy.dtype=float32 is deliberate: lerobot_train derives Accelerator
# mixed_precision from policy.dtype (the accelerate-launch flag is ignored),
# and dtype=bfloat16 would put weights AND AdamW moments in bf16 with no fp32
# master copy -- at lr 2.5e-5 most updates round to zero (silent frozen net).
# fp32 statics ~55GB/GPU (weights 14.5 + grads 13.4 + AdamW 26.8) fit the
# 96GB RTX PRO 6000s with gradient checkpointing at per-GPU batch 8.
# rename_map maps YAM cameras onto the base checkpoint's openpi camera slots.

set -euo pipefail
source /n/netscratch/ydu_lab/Lab/asethi/molmoact-ft/env.sh   # tokens + wandb creds
export HL=/n/holylabs/kempner_ydu_lab/Lab/asethi
export HF_HOME=$HL/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# wandb ARTIFACT STAGING follows WANDB_DATA_DIR/XDG data home (NOT WANDB_DIR) —
# the profile's XDG points at netscratch, whose quota kills 14.5GB stagings
# (this crashed job 37421128 at the step-7000 checkpoint's log_policy call).
export WANDB_DATA_DIR=/tmp/wandb-data-$USER
export WANDB_ARTIFACT_DIR=/tmp/wandb-artifacts-$USER

export VENV=/n/netscratch/ydu_lab/Lab/asethi/venvs/lerobot-aq
# fixedstats overlay: symlinked data/videos + meta/stats.json with EXACT global
# q01/q99 (stock stats carry upstream bug #4156 — per-episode weighted-mean
# aggregation — leaving 18.6% of values outside [-1,1] after normalization)
export DATA_ROOT=$HL/molmoact-ft/data/yam-vive-teleop-fixedstats
export NNODES=$SLURM_JOB_NUM_NODES
export WORLD=$((NNODES * ${GPUS_PER_NODE:-8}))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
export TFS_JSON=$(cat $HL/molmoact-ft/tfs_run2.json)
export RENAME_MAP='{"observation.images.top":"observation.images.base_0_rgb","observation.images.left":"observation.images.left_wrist_0_rgb","observation.images.right":"observation.images.right_wrist_0_rgb"}'
LOGFILE=$HL/molmoact-ft/logs/pi05-$SLURM_JOB_ID.out

TRAIN_STEPS=12000
export TRAIN_STEPS
echo "nodes=$NNODES world=$WORLD steps=$TRAIN_STEPS branch=$(git -C ~/lerobot-aq rev-parse --abbrev-ref HEAD)@$(git -C ~/lerobot-aq rev-parse --short HEAD)"

run_train() {
  # env args: RUN_STEPS RUN_WARMUP RUN_SAVE RUN_LOGF RUN_OUT RUN_NAME RUN_WANDB
  if [ -e "$RUN_OUT/checkpoints/last" ]; then
    echo "resuming from $RUN_OUT/checkpoints/last"
    export RUN_RESUME="--resume=true --config_path=$RUN_OUT/checkpoints/last/pretrained_model/train_config.json"
    # NO --policy.path on resume: _resolve_pretrained_from_cli's policy-path branch
    # preempts the resume branch (checkpoint_path stays None -> load_training_state
    # crashes) and would reload BASE weights instead of the checkpoint's.
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
    --dataset.repo_id=brandonyang/yam-vive-teleop \
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
    --steps=$RUN_STEPS --batch_size=8 --seed=1000 \
    --num_workers=8 --dataloader_multiprocessing_context=spawn \
    --tolerance_s=0.0001 \
    --log_freq=$RUN_LOGF --save_checkpoint=true --save_freq=$RUN_SAVE \
    --env_eval_freq=0 --eval_steps=0 \
    --wandb.disable_artifact=true \
    $RUN_WANDB \
    --job_name=$RUN_NAME --output_dir=$RUN_OUT
  '
}

# Hard gate: pi05 from_pretrained swallows weight-load failures and silently
# returns random init (modeling_pi05.py:810-813). Refuse to continue if the
# load-success marker is missing from this job's log.
check_weights_loaded() {
  if grep -q "Returning model without loading pretrained weights" "$LOGFILE" \
     || grep -q "Warning: Could not load state dict" "$LOGFILE"; then
    echo "FATAL: pretrained weight load failed (random init detected)"; exit 1
  fi
  grep -q "Loaded state dict from model.safetensors" "$LOGFILE" \
    || { echo "FATAL: no weight-load confirmation in log"; exit 1; }
  echo "WEIGHTS_LOADED_OK"
}

# MAIN_OUT_OVERRIDE lets a RESUBMITTED job (new job id) adopt a previous run dir
# and resume from its checkpoints/last instead of starting a fresh run.
MAIN_OUT=${MAIN_OUT_OVERRIDE:-$HL/molmoact-ft/runs/pi05-12k-${WORLD}gpu-$SLURM_JOB_ID}

# ---- Phase 1: 25-step canary (skipped on requeue once main run has checkpoints) ----
if [ ! -e "$MAIN_OUT/checkpoints" ]; then
  export RUN_STEPS=25 RUN_WARMUP=5 RUN_SAVE=25 RUN_LOGF=5 RUN_WANDB="--wandb.enable=false"
  export RUN_OUT=$HL/molmoact-ft/runs/canary-pi05-$SLURM_JOB_ID RUN_NAME=canary-pi05-$SLURM_JOB_ID
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
export RUN_NAME=pi05-12k
echo "=== TRAIN START $(date) ==="
run_train
check_weights_loaded
echo "=== TRAIN_OK $(date) ==="
