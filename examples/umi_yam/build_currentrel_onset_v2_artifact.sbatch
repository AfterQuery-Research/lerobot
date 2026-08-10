#!/usr/bin/env bash
#SBATCH --job-name=umi-onset-v2-build
#SBATCH --partition=serial_requeue
#SBATCH --account=ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/umi-onset-v2-build-%j.out

set -euo pipefail

: "${UMI_YAM_CODE_ROOT:?set the immutable code snapshot root}"
: "${UMI_YAM_CODE_MANIFEST_SHA256:?pin the snapshot manifest SHA-256}"
: "${UMI_YAM_LAB_ROOT:?set the Kempner lab storage root}"
: "${UMI_YAM_VENV:?set the tested Python 3.12 environment}"

CODE_ROOT=$UMI_YAM_CODE_ROOT
CODE_MANIFEST=$CODE_ROOT/.umi_yam_snapshot.sha256
SOURCE_ROOT=$UMI_YAM_LAB_ROOT/molmoact-ft/data/dual-lidar-umi-filtered-ad1db
DATA_PARENT=$UMI_YAM_LAB_ROOT/molmoact-ft/data
FINAL_ROOT=$DATA_PARENT/dual-lidar-umi-currentrel-r6d-onset-v2
STAGING_ROOT=$DATA_PARENT/.dual-lidar-umi-currentrel-r6d-onset-v2.staging-$SLURM_JOB_ID

test -d "$CODE_ROOT"
test ! -w "$CODE_ROOT"
test -f "$CODE_MANIFEST"
test "$(sha256sum "$CODE_MANIFEST" | awk '{print $1}')" = "$UMI_YAM_CODE_MANIFEST_SHA256"
(cd "$CODE_ROOT" && sha256sum --quiet --strict -c .umi_yam_snapshot.sha256)

test -d "$SOURCE_ROOT"
test "$(sha256sum "$SOURCE_ROOT/meta/info.json" | awk '{print $1}')" = \
  252cb2f9503383716e19f89f199036a08ebd0c6f20be6ccf80d0bc32221ff3f8
test "$(sha256sum "$SOURCE_ROOT/meta/filtering.json" | awk '{print $1}')" = \
  f4cc85914f889ffb5b9cf82622cff7051c9f18683dc260416e184206ca430f5e
test ! -e "$FINAL_ROOT"
test ! -e "$STAGING_ROOT"

export PYTHONPATH=$CODE_ROOT/src
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$UMI_YAM_VENV/bin/python" \
  -m lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v2 \
  --source-root="$SOURCE_ROOT" \
  --output-root="$STAGING_ROOT"

"$UMI_YAM_VENV/bin/python" \
  "$CODE_ROOT/examples/umi_yam/validate_currentrel_onset_v2_inputs.py" \
  "$STAGING_ROOT"
(cd "$STAGING_ROOT" && sha256sum --quiet --strict -c meta/artifact_manifest.sha256)

# Publish only after conversion and both independent integrity checks pass.
chmod -R a-w "$STAGING_ROOT"
mv -T -- "$STAGING_ROOT" "$FINAL_ROOT"

"$UMI_YAM_VENV/bin/python" \
  "$CODE_ROOT/examples/umi_yam/validate_currentrel_onset_v2_inputs.py" \
  "$FINAL_ROOT"
(cd "$FINAL_ROOT" && sha256sum --quiet --strict -c meta/artifact_manifest.sha256)
(cd "$CODE_ROOT" && sha256sum --quiet --strict -c .umi_yam_snapshot.sha256)

printf 'ONSET_V2_ARTIFACT_OK root=%s manifest_sha256=%s code_manifest_sha256=%s\n' \
  "$FINAL_ROOT" \
  "$(sha256sum "$FINAL_ROOT/meta/artifact_manifest.sha256" | awk '{print $1}')" \
  "$UMI_YAM_CODE_MANIFEST_SHA256"
