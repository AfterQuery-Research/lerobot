# π0.5 fine-tuning on the bimanual YAM ("Put all oranges in the bowl")

End-to-end recipe used to produce the three published π0.5 checkpoints
(2026-08-05/06, FASRC Kempner cluster, 8×RTX PRO 6000 Blackwell 96GB):

| Model (HF, public) | Dataset | Cameras |
|---|---|---|
| `ASethi04/pi05-BimanualYAM-oranges` | `brandonyang/yam-vive-teleop` | top + L/R RealSense |
| `ASethi04/pi05-BimanualYAM-oranges-uw-top` | `brandonyang/yam-ultrawide-teleop` | top + ultrawide wrists |
| `ASethi04/pi05-BimanualYAM-oranges-uw-notop` | `brandonyang/yam-ultrawide-teleop` | ultrawide wrists only |

Shared recipe: `--policy.path=lerobot/pi05_base`, 12,000 steps, per-GPU batch 8
(global 64), seed 1000, chunk 30/30, **fp32** (see below), AdamW 2.5e-5 cosine →
2.5e-6, warmup 600, quantile normalization with **corrected global q01/q99**,
narrowed color jitter + affine augmentation, gradient checkpointing.

## Files

- `train_pi05.sbatch` — RealSense run (fixed dataset root). Canary phase (25 steps
  in the same allocation), weights-loaded hard gate, requeue+resume.
- `train_pi05_v2.sbatch` — parameterized version (env: `RUN_TAG`, `DATA_ROOT_V2`,
  `RENAME_MAP_V2`, `BS`, `MAIN_OUT_OVERRIDE`) used for both ultrawide variants.
- `preflight_pi05_light.py` / `preflight_pi05_uw.py` — login-node preflights:
  exact-argv draccus parse (`--policy.path` merge via sys.argv), dataset load,
  stats verification, preprocessor batch. (Login nodes OOM-kill the 14.5GB CPU
  weight load; the GPU canary covers that.)
- `eval_openloop_pi05.py` + `eval_pi05_12k.sbatch` — open-loop 30-step chunk L1,
  degeneracy/limit/jerk gates, camera-shuffle probe. In-sample by design (100/0
  split): a brokenness gate, not a success predictor.
- `probe_suite_pi05.py` + `probe_suite.sbatch` — five deeper offline probes,
  policy-agnostic (works on pi05 and MolmoAct2 checkpoints): grasp-event timing
  (hysteresis 0.78/0.82, clean-window guard), chunk-stitching consistency,
  hue-rotation counterfactual with brightness control, state/vision ablations,
  task-string sensitivity, plus a seeded identity-rerun noise-floor control.
- `PI05-SERVING-NOTES.md` — deployment flags and the serving fixes required for
  pi05 checkpoints (see PR "fix(remote_inference): serve pi05 fine-tune
  checkpoints correctly").

Paths inside the sbatch files are FASRC-specific (holylabs storage, netscratch
venv) — adapt `HL`/`VENV`/partitions to your cluster.

## Verified gotchas (each cost us a failed run or a silent near-miss)

1. **fp32 is mandatory for full fine-tuning here.** `lerobot_train` derives
   Accelerator mixed-precision from `--policy.dtype` (the `accelerate launch
   --mixed_precision` flag is ignored), and `dtype=bfloat16` puts AdamW moments
   in bf16 with no fp32 master weights: at lr 2.5e-5 most updates round to zero
   while the loss still falls. fp32 statics ≈55GB/GPU; batch 8 + grad
   checkpointing ≈ 79.4GB on 96GB cards, ~2.65 s/step.
2. **`--policy.type=pi05` alone trains from RANDOM init.** Weights only load via
   `--policy.path`. And `PI05Policy.from_pretrained` swallows load failures,
   returning random weights with only a printed warning — the sbatch greps the
   log for `Loaded state dict from model.safetensors` and aborts otherwise.
3. **Camera slots need a top-level `--rename_map`** onto the base checkpoint's
   `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`. Omitting a slot (no-top
   variant) engages pi05's native missing-camera masking (−1 pad + zero attn
   mask — in-distribution with openpi pretraining; do NOT feed black frames).
4. **Recompute dataset q01/q99 globally before training** (upstream #4156:
   per-episode weighted-mean aggregation biases quantiles inward — 18.6% of
   values landed outside [−1,1] on yam-vive-teleop, ~50% q-range bias on
   yam-ultrawide). We train against an overlay root: symlinked data/videos,
   copied meta with exact global quantiles.
5. **Never combine `--policy.path` with `--resume=true`** (see PR
   "fix(configs): reject --policy.path combined with --resume=true").
6. **wandb artifact staging follows `WANDB_DATA_DIR`/XDG, not `WANDB_DIR`** —
   point it at node-local disk and pass `--wandb.disable_artifact=true`, or a
   full staging filesystem kills the run at a checkpoint save.
7. `--policy.compile_model` must stay off (upstream #4178: NaN loss / CUDA
   crash, fix unmerged). No EMA exists in lerobot training.

## Offline probe findings (7-model matrix, details in the run logs)

- π0.5 models are far more visually grounded than MolmoAct2 on the same data
  (black-image error ratio ~7× vs 1.6–3×) and have the best grasp-event timing
  (≤1.1 ticks mean, p90 ≤3 ticks at 30Hz).
- Dropping the top camera costs nothing measurable offline for either family.
- Calibration caveat: probes only weakly separated a hardware-validated
  better/worse MolmoAct2 pair (run2 vs run3) — trust large probe gaps, not
  small ones, and let hardware arbitrate.
