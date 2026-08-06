#!/bin/bash
#SBATCH -A ydu_lab
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -o /n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/logs/%x-%j.out
# Runs probe_suite_pi05.py over MULTIPLE checkpoints sequentially on one GPU.
# env: PROBE_CKPTS (space-separated "name=ckpt_path" pairs), PROBE_ROOT,
#      PROBE_REPO_ID, PROBE_EPISODES, out files land as probe_<name>.json

set -uo pipefail
source /n/netscratch/ydu_lab/Lab/asethi/molmoact-ft/env.sh
export HL=/n/holylabs/kempner_ydu_lab/Lab/asethi
export HF_HOME=$HL/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

FAIL=0
for pair in ${PROBE_CKPTS:?set PROBE_CKPTS}; do
  name="${pair%%=*}"; ckpt="${pair#*=}"
  echo "=== PROBE $name ($ckpt) $(date) ==="
  $NS/venvs/lerobot-aq/bin/python -u $HL/molmoact-ft/probe_suite_pi05.py \
    --ckpt "$ckpt" \
    --root ${PROBE_ROOT:?} \
    --repo_id ${PROBE_REPO_ID:?} \
    --episodes ${PROBE_EPISODES:?} \
    --stride 60 \
    --out $HL/molmoact-ft/probe_${name}.json || { echo "PROBE_FAIL $name"; FAIL=1; }
done
[ $FAIL -eq 0 ] && echo "=== PROBES_ALL_DONE $(date) ===" || echo "=== PROBES_DONE_WITH_FAILURES $(date) ==="
