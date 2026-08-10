#!/bin/bash
#SBATCH --job-name=umi-onset-v3-12k
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:7
#SBATCH --cpus-per-task=84
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [[ -n ${UMI_YAM_ENV_FILE:-} ]]; then
  [[ -r "$UMI_YAM_ENV_FILE" ]] || {
    echo "UMI_YAM_ENV_FILE is not readable: $UMI_YAM_ENV_FILE" >&2
    exit 2
  }
  # shellcheck disable=SC1090
  source "$UMI_YAM_ENV_FILE"
fi

: "${UMI_YAM_LAB_ROOT:?set UMI_YAM_LAB_ROOT to the Kempner lab-storage user root}"
: "${UMI_YAM_SCRATCH_ROOT:?set UMI_YAM_SCRATCH_ROOT to the scratch-storage user root}"
: "${UMI_YAM_VENV:?set UMI_YAM_VENV to the proven MolmoAct2 training environment}"
: "${UMI_YAM_CODE_ROOT:?point UMI_YAM_CODE_ROOT at the read-only onset-v3 code snapshot}"
: "${UMI_YAM_CODE_MANIFEST_SHA256:?pin the snapshot manifest file SHA-256}"
: "${UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256:?pin the dataset checksum-manifest SHA-256}"
: "${UMI_YAM_ONSET_V3_GATE_PLAN_SHA256:?pin the checkpoint-gate plan SHA-256}"
: "${UMI_YAM_SMOKE_JOB_ID:?full training must name its successful smoke dependency}"
: "${WANDB_API_KEY:?export WANDB_API_KEY from the mode-600 UMI_YAM_ENV_FILE}"

HL=$UMI_YAM_LAB_ROOT
SCRATCH=$UMI_YAM_SCRATCH_ROOT
VENV=$UMI_YAM_VENV
CODE_ROOT=$UMI_YAM_CODE_ROOT
DATA_ROOT=$HL/molmoact-ft/data/dual-lidar-umi-currentrel-r6d-onset-v3
MODEL_ROOT=$HL/molmoact-ft/models/MolmoAct2-e432d85
MODEL_REV=e432d85f6e039edca44afb93c262f3084ab72a9c
CODE_MANIFEST=$CODE_ROOT/.umi_yam_snapshot.sha256
DATA_MANIFEST=$DATA_ROOT/meta/artifact_manifest.sha256
GATE_PLAN=$CODE_ROOT/examples/umi_yam/onset_v3_checkpoint_gate.json
RUN_OUT=$SCRATCH/molmoact-ft/runs/molmoact2-umi-currentrel-r6d-onset-v3-12k
INNER_LOG=$SCRATCH/molmoact-ft/logs/molmoact2-umi-currentrel-r6d-onset-v3-12k-inner-$SLURM_JOB_ID.log
SMOKE_REPORT=$SCRATCH/molmoact-ft/validation/onset-v3-smoke-$UMI_YAM_SMOKE_JOB_ID.json
REPORT=$SCRATCH/molmoact-ft/validation/onset-v3-full-$SLURM_JOB_ID.json

export HF_HOME=$HL/hf-cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH=$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=online
export WANDB_DIR=$SCRATCH/molmoact-ft/wandb
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))

test -x "$VENV/bin/accelerate"
test -x "$VENV/bin/python"
mkdir -p "$WANDB_DIR"
test -f "$CODE_ROOT/examples/umi_yam/validate_currentrel_onset_v3_inputs.py"
test -f "$CODE_ROOT/examples/umi_yam/validate_currentrel_onset_v3_training_run.py"
test -f "$GATE_PLAN"
test -f "$CODE_MANIFEST"
test ! -w "$CODE_ROOT"
test -f "$DATA_MANIFEST"
test -f "$MODEL_ROOT/model.safetensors.index.json"
test "$(git -C "$MODEL_ROOT" rev-parse HEAD)" = "$MODEL_REV"
test -z "$(git -C "$MODEL_ROOT" status --porcelain)"
test ! -e "$RUN_OUT"
test ! -e "$INNER_LOG"
test ! -e "$REPORT"

test "$(sha256sum "$CODE_MANIFEST" | awk '{print $1}')" = "$UMI_YAM_CODE_MANIFEST_SHA256"
(cd "$CODE_ROOT" && sha256sum --quiet --strict -c .umi_yam_snapshot.sha256)
test "$(sha256sum "$DATA_MANIFEST" | awk '{print $1}')" = \
  "$UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256"
test "$(sha256sum "$GATE_PLAN" | awk '{print $1}')" = "$UMI_YAM_ONSET_V3_GATE_PLAN_SHA256"
(cd "$DATA_ROOT" && sha256sum --quiet --strict -c meta/artifact_manifest.sha256)

cd "$CODE_ROOT"
"$VENV/bin/python" examples/umi_yam/validate_currentrel_onset_v3_inputs.py "$DATA_ROOT"
"$VENV/bin/python" examples/umi_yam/validate_currentrel_onset_v3_training_run.py verify-smoke \
  --report="$SMOKE_REPORT" \
  --job-id="$UMI_YAM_SMOKE_JOB_ID" \
  --dataset-manifest-sha256="$UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256" \
  --code-manifest-sha256="$UMI_YAM_CODE_MANIFEST_SHA256" \
  --gate-plan-sha256="$UMI_YAM_ONSET_V3_GATE_PLAN_SHA256"
TFS_JSON=$("$VENV/bin/python" -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' \
  examples/umi_yam/molmoact2_currentrel_color_v1.json)

echo "FULL_START job=$SLURM_JOB_ID smoke=$UMI_YAM_SMOKE_JOB_ID model=allenai/MolmoAct2@$MODEL_REV code_root=$CODE_ROOT" | \
  tee "$INNER_LOG"
set +e
srun --kill-on-bad-exit=1 "$VENV/bin/accelerate" launch \
  --multi_gpu --num_processes=7 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
  --machine_rank=0 --main_process_ip="$MASTER_ADDR" --same_network \
  --main_process_port="$MASTER_PORT" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=brandonyang/dual-lidar-umi-currentrel-r6d-onset-v3 \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.037037037037037035 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs="$TFS_JSON" \
  --policy.type=molmoact2 \
  --policy.checkpoint_path="$MODEL_ROOT" \
  --policy.checkpoint_revision="$MODEL_REV" \
  --policy.device=cuda \
  --policy.action_mode=continuous \
  --policy.inference_action_mode=continuous \
  --policy.chunk_size=24 \
  --policy.n_action_steps=24 \
  --policy.n_obs_steps=1 \
  --policy.num_flow_timesteps=8 \
  --policy.expected_max_action_dim=32 \
  --policy.mask_action_dim_padding=true \
  --policy.normalize_gripper=false \
  --policy.model_dtype=bfloat16 \
  --policy.llm_residual_dropout=0.1 \
  --policy.train_mode_vlm=lora \
  --policy.lora_rank=64 \
  --policy.lora_alpha=16 \
  --policy.lora_dropout=0.05 \
  --policy.freeze_embedding=true \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_vit_lr=5e-6 \
  --policy.optimizer_connector_lr=5e-6 \
  --policy.optimizer_action_expert_lr=5e-5 \
  --policy.scheduler_warmup_steps=600 \
  --policy.scheduler_decay_steps=12000 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.setup_type="bimanual UMI grippers mounted on YAM" \
  --policy.control_mode="current-relative dual end-effector trajectory" \
  --policy.image_keys='["observation.images.umi1","observation.images.umi2"]' \
  --policy.push_to_hub=false \
  --steps=12000 \
  --batch_size=4 \
  --seed=1000 \
  --num_workers=8 \
  --dataloader_multiprocessing_context=spawn \
  --tolerance_s=0.0001 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --env_eval_freq=0 \
  --eval_steps=1000 \
  --max_eval_samples=0 \
  --wandb.enable=true \
  --wandb.entity=aq-robotics \
  --wandb.project=molmoact2-yam-oranges \
  --wandb.disable_artifact=true \
  --job_name=molmoact2-umi-currentrel-r6d-onset-v3-12k \
  --output_dir="$RUN_OUT" 2>&1 | tee -a "$INNER_LOG"
trainer_rc=${PIPESTATUS[0]}
set -e
if ((trainer_rc != 0)); then
  echo "FULL_FAILED trainer_exit=$trainer_rc"
  exit "$trainer_rc"
fi

# Re-run both frozen-input checks before accepting any checkpoint artifacts.
(cd "$CODE_ROOT" && sha256sum --quiet --strict -c .umi_yam_snapshot.sha256)
(cd "$DATA_ROOT" && sha256sum --quiet --strict -c meta/artifact_manifest.sha256)
"$VENV/bin/python" examples/umi_yam/validate_currentrel_onset_v3_inputs.py "$DATA_ROOT"
"$VENV/bin/python" examples/umi_yam/validate_currentrel_onset_v3_training_run.py full \
  --dataset-root="$DATA_ROOT" \
  --dataset-manifest-sha256="$UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256" \
  --code-manifest-sha256="$UMI_YAM_CODE_MANIFEST_SHA256" \
  --gate-plan="$GATE_PLAN" \
  --gate-plan-sha256="$UMI_YAM_ONSET_V3_GATE_PLAN_SHA256" \
  --model-root="$MODEL_ROOT" \
  --job-id="$SLURM_JOB_ID" \
  --run-root="$RUN_OUT" \
  --log="$INNER_LOG" \
  --report="$REPORT"

echo "FULL_OK job=$SLURM_JOB_ID output=$RUN_OUT report=$REPORT"
