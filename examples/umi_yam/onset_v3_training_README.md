# Internal onset-v3 MolmoAct2 launch recipe

Status: the reference onset-v3 run completed all 12,000 updates and produced checkpoints at every
1,000-step boundary. The original batch job exited nonzero only because its post-run validator
expected a stale normalizer tensor count. The corrected validator in this branch checks the exact
86-tensor semantic schema and has validated the complete, unmodified checkpoint tree. This is a
training result, not hardware approval; checkpoint advancement still requires the separate offline
IK/behavior and temporal gates described below.

## Frozen experiment contract

- Dataset root:
  `$UMI_YAM_LAB_ROOT/molmoact-ft/data/dual-lidar-umi-currentrel-r6d-onset-v3`
- Dataset/repo ID: `brandonyang/dual-lidar-umi-currentrel-r6d-onset-v3`
- Schema sidecar: `meta/umi_current_relative_r6d_onset_v3.json`
- Split: episodes `0..51` train and `52,53` full holdout. No episode may change split.
- Locked artifact counts: `47,237` train frames, `1,884` full-holdout frames, `49,121` total.
  Episodes `0`, `40`, and `46` receive the predeclared source-exclusive tail cuts; the two
  holdouts are unchanged. No universal tail trim or additional trajectory smoothing is permitted.
- A `k=1..24` chunk has exactly `300` padded future positions per episode. Statistics and loss
  must exclude them: `1,118,088` valid train action rows, `44,616` valid holdout rows, and
  `1,162,704` valid rows over all 54 episodes. The padded-inclusive train count is `1,133,688` and
  must not appear in the checkpoint action normalizer.
- Target: current-query-anchored `24 x 20` continuous dual-arm SE(3) R6D + gripper target.
- Base: local clean checkout of `allenai/MolmoAct2` at
  `e432d85f6e039edca44afb93c262f3084ab72a9c`.
- Augmentation: the existing brightness/contrast/saturation ColorJitter contract in
  `molmoact2_currentrel_color_v1.json`.
- Optimizer: the proven learning rates (`5e-5` VLM/action expert, `5e-6` vision/connector),
  warmup `600`, cosine decay through update `12000`, terminal learning rate `1e-6`.
- VLM residual dropout: `0.1`, set explicitly by both launch recipes (the library default remains
  the backward-compatible `0.0`).
- Full run: one RTX PRO 6000 Blackwell node, seven GPUs, per-rank batch `4`, global batch `28`,
  BF16, seed `1000`, exactly `12000` updates.
- Full holdout evaluation and checkpoint save: exactly steps
  `1000,2000,...,12000`; `max_eval_samples=0` means both held-out episodes are evaluated in full.
  Evaluation has no ColorJitter. The optimizer, `12000`-update, warmup/decay, and checkpoint/eval
  cadence are inherited from successful v1 job `37921247`, which completed in about `3 h 50 min`
  on eight GPUs and whose 12 checkpoints occupied `168 GiB`. The seven-GPU global batch is
  intentionally different: it presents `336,000` samples, about `7.113` train-frame epochs,
  versus v1's `384,000` presentations, about `8.129` epochs. The 12-hour request and unchanged
  checkpoint storage cadence remain conservative; onset alignment, targeted suffix removal,
  terminal-target masking, and the lower sample exposure are explicit v3 differences.

The one-GPU smoke keeps the same per-rank batch and full-run scheduler values but runs only five
updates with no evaluation/checkpoint. It tests data/model/scheduler wiring and finite gradients;
it is not a learning-quality result.

The smoke uses only immutable local logs. Full training additionally reports to the W&B entity
`aq-robotics`, project `molmoact2-yam-oranges`; checkpoint artifacts remain disabled there because
the immutable local checkpoint tree is authoritative.

## Freeze inputs

After conversion finishes, validate and pin the artifact. The converter copies all 108 source MP4s
byte-for-byte into the artifact. Its GNU checksum manifest covers every parquet, metadata, video,
and README file; the independent validator also verifies the video path set and each copy against
the pinned source bytes.

```bash
export UMI_YAM_LAB_ROOT=/path/to/persistent-storage
export UMI_YAM_SCRATCH_ROOT=/path/to/scratch-storage
export UMI_YAM_VENV=/path/to/tested-python-environment

DATA_ROOT=$UMI_YAM_LAB_ROOT/molmoact-ft/data/dual-lidar-umi-currentrel-r6d-onset-v3
"$UMI_YAM_VENV/bin/python" \
  examples/umi_yam/validate_currentrel_onset_v3_inputs.py "$DATA_ROOT"
(cd "$DATA_ROOT" && sha256sum --quiet --strict -c meta/artifact_manifest.sha256)
export UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256=$(
  sha256sum "$DATA_ROOT/meta/artifact_manifest.sha256" | awk '{print $1}'
)
```

Freeze the tested working tree only after all onset-v3 changes and tests are final. This matches the
read-only snapshot pattern used by the successful v1 run, but adds a checked content manifest.

```bash
SOURCE_ROOT=/path/to/this/checkout
SNAPSHOT_ROOT=$UMI_YAM_LAB_ROOT/molmoact-ft/code/umi-yam-onset-v3-training-$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "$SNAPSHOT_ROOT"
mkdir -p "$SNAPSHOT_ROOT"
rsync -a --no-group \
  --exclude=.git \
  --exclude=.venv/ \
  --exclude='__pycache__/' \
  --exclude=.pytest_cache/ \
  --exclude=.ruff_cache/ \
  "$SOURCE_ROOT/" "$SNAPSHOT_ROOT/"
(
  cd "$SNAPSHOT_ROOT"
  find . -xdev -type f ! -name .umi_yam_snapshot.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > .umi_yam_snapshot.sha256
  sha256sum --quiet --strict -c .umi_yam_snapshot.sha256
)
chmod -R a-w "$SNAPSHOT_ROOT"
export UMI_YAM_CODE_ROOT=$SNAPSHOT_ROOT
export UMI_YAM_CODE_MANIFEST_SHA256=$(
  sha256sum "$SNAPSHOT_ROOT/.umi_yam_snapshot.sha256" | awk '{print $1}'
)
test ! -w "$UMI_YAM_CODE_ROOT"
export UMI_YAM_ONSET_V3_GATE_PLAN_SHA256=$(
  sha256sum "$UMI_YAM_CODE_ROOT/examples/umi_yam/onset_v3_checkpoint_gate.json" | awk '{print $1}'
)
```

Put the W&B API key only in a non-versioned, mode-600 environment file and set
`UMI_YAM_ENV_FILE`. The file must export `WANDB_API_KEY`; neither snapshot nor dataset may contain
credentials.

## Submit smoke, then the dependent full run

This is the exact submission chain; run it only after the freeze checks above. It prevents the full
job from starting unless the smoke exits zero and leaves a validation report bound to the same code
and dataset manifests.

```bash
JOB_ROOT=$UMI_YAM_SCRATCH_ROOT/molmoact-ft
mkdir -p "$JOB_ROOT/logs" "$JOB_ROOT/validation"
SBATCH_EXPORT=ALL,UMI_YAM_LAB_ROOT,UMI_YAM_SCRATCH_ROOT,UMI_YAM_VENV,UMI_YAM_CODE_ROOT,UMI_YAM_CODE_MANIFEST_SHA256,UMI_YAM_ONSET_V3_ARTIFACT_MANIFEST_SHA256,UMI_YAM_ONSET_V3_GATE_PLAN_SHA256
if [[ -n ${UMI_YAM_ENV_FILE:-} ]]; then
  SBATCH_EXPORT+=,UMI_YAM_ENV_FILE
fi

SMOKE_SUBMIT=$(sbatch --parsable \
  --chdir="$JOB_ROOT" \
  --export="$SBATCH_EXPORT" \
  "$UMI_YAM_CODE_ROOT/examples/umi_yam/train_currentrel_onset_v3_smoke.sbatch")
SMOKE_ID=${SMOKE_SUBMIT%%;*}

FULL_SUBMIT=$(sbatch --parsable \
  --dependency="afterok:$SMOKE_ID" \
  --kill-on-invalid-dep=yes \
  --chdir="$JOB_ROOT" \
  --export="$SBATCH_EXPORT,UMI_YAM_SMOKE_JOB_ID=$SMOKE_ID" \
  "$UMI_YAM_CODE_ROOT/examples/umi_yam/train_currentrel_onset_v3_full.sbatch")
FULL_ID=${FULL_SUBMIT%%;*}
printf 'smoke=%s full=%s\n' "$SMOKE_ID" "$FULL_ID"
```

Do not create the full output directory in advance: the full job intentionally rejects any existing
`molmoact2-umi-currentrel-r6d-onset-v3-12k` run to prevent overwrite/resume ambiguity.

Monitor without changing either job:

```bash
squeue -j "$SMOKE_ID,$FULL_ID" -o '%.18i %.9P %.28j %.2t %.10M %.10L %R'
tail -F "$JOB_ROOT/logs/umi-onset-v3-smoke-$SMOKE_ID.out"
tail -F "$JOB_ROOT/logs/umi-onset-v3-12k-$FULL_ID.out"
```

Expected validation reports:

```text
$JOB_ROOT/validation/onset-v3-smoke-$SMOKE_ID.json
$JOB_ROOT/validation/onset-v3-full-$FULL_ID.json
```

The smoke validator requires finite loss, action-flow loss, and gradient norm at exactly steps
`1..5`. The full validator requires finite values at all 1,200 logged training updates; one finite
full-holdout loss at each of the 12 checkpoint steps; exactly 12 checkpoint directories; the
correct final symlink; exact model/optimizer safetensors size and tensor counts; intact processor,
normalizer, RNG, optimizer, and scheduler artifacts; the exact seven-rank/per-rank-batch training
state; the valid-only action-normalizer count of `1,118,088`; and the frozen
dataset/model/config/ColorJitter/gate-plan contract. Both
jobs recheck the code and dataset manifests after training so input mutation fails the job.

## Predeclared checkpoint selection

Training loss alone never advances a checkpoint. The exact plan is
`onset_v3_checkpoint_gate.json`. Run its independent model-to-resolver-to-strict-IK gate on
**all 12** candidates and all ten cells: episodes `52,53` x seeds `0..4`, one fresh query per
cell, ten inference flow steps, and only the first 15 predicted rows executed. Do not execute row
16..24 for eligibility, stop after a favorable checkpoint/seed, or replace the start anchor.

The frozen simulated/measured anchor is the exact 14-D absolute driver target
`[0,.05,.05,0,0,0,1, 0,.05,.05,0,0,0,1]`. Ground-truth audit passes all `1,620/1,620`
first-15 arm rows and all `2,592/2,592` full-24 diagnostic arm rows there, with at most four
settling dispatches. Canonical q0 is retained only as negative provenance (`1,614/1,620`
first-15 arm rows and `2,532/2,592` full-24 arm rows). The interior anchor is still
`hardware_start_verified=false`: it needs supervised collision and homologous jaw/wrist-camera
viewpoint confirmation before any physical rollout, and this plan does not alter runtime config.

Offline eligibility requires both (a) all ten first-15 strict execution cells and (b) at least a
10% reduction from identity in the predeclared physical pose metric. That metric converts R6D to
SO(3), uses translation error in meters scaled by 1 cm and geodesic rotation error scaled by 10
degrees, and pools all 10 cells x 2 arms x 15 rows without exclusions. The aggregate must also
report predicted/target translation and rotation magnitudes plus endpoint direction cosine for
each arm and cell. This replaces the old weak test where a tiny pooled component-L1 improvement
could admit a reachable but nearly static prediction.

The runnable stage-one and temporal gates, independent aggregators, simulator diagnostics, remote
policy server, and robot bridge live in the separate deployment PR. Keeping those files out of this
branch makes the training diff independently reviewable and prevents cluster- or robot-specific
runtime code from becoming an undeclared training dependency. That deployment stack must bind the
training report, every checkpoint hash, this JSON contract, the dataset manifest, the frozen code
snapshot, the exact kinematics checkout, and its own source hashes.

Final selection is intentionally two-stage: every checkpoint that passes the fixed ten-cell onset
gate must also pass the complete teacher-forced temporal gate. Only then may immutable full-holdout
loss choose among passing checkpoints. Neither a lower IK residual nor a visually favorable replay
may override this rule. If no checkpoint passes both gates, nothing is published as a deployment
candidate.

The gates remain diagnostic. They do not prove autonomous visual recovery, orange or bowl contact,
friction, grasp success, collision clearance, calibrated gripper actuation, or hardware safety.
The deployment PR therefore keeps `hardware_start_verified=false` and hardware control disabled
until the hardware team supplies and verifies the physical start, rig, camera, CAN, and gripper
evidence described there.
