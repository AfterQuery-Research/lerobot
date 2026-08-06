#!/bin/bash
#SBATCH -A ydu_lab
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -o /n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/logs/%x-%j.out
# Offline open-loop eval of the realsense pi05-12k run. Two modes via env:
#   EVAL_DIR  checkpoint glob dir (gate: eval_gate_ckpts w/ only 012000; curve: checkpoints)
#   EVAL_OUT  output json
# Submit gate:  sbatch -J eval-pi05-gate -t 1:00:00 --export=ALL eval_pi05_12k.sbatch  (EVAL_DIR=eval_gate_ckpts)
# Submit curve: sbatch -J eval-pi05-curve -t 6:00:00 --export=ALL eval_pi05_12k.sbatch (EVAL_DIR=checkpoints)

set -uo pipefail
source /n/netscratch/ydu_lab/Lab/asethi/molmoact-ft/env.sh
export HL=/n/holylabs/kempner_ydu_lab/Lab/asethi
export HF_HOME=$HL/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN=${RUN_DIR:-$HL/molmoact-ft/runs/pi05-12k-8gpu-37421128}
DATA=${EVAL_DATA_ROOT:-$HL/molmoact-ft/data/yam-vive-teleop-fixedstats}
REPO_ID=${EVAL_REPO_ID:-brandonyang/yam-vive-teleop}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$NS/venvs/lerobot-aq/bin/python -u $HL/molmoact-ft/eval_openloop_pi05.py \
  --run_dir $RUN/${EVAL_DIR:?set EVAL_DIR} \
  --root $DATA \
  --repo_id $REPO_ID \
  --episodes ${EVAL_EPISODES:-3 17 41 62 78} \
  --stride 60 \
  --batch_size 4 \
  --num_workers 4 \
  --camera_probe \
  --out $HL/molmoact-ft/${EVAL_OUT:?set EVAL_OUT}

echo "=== EVAL_PI05_DONE $(date) ==="
