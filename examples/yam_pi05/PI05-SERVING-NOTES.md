# Deploying pi05 YAM checkpoints (audit 2026-08-06)

Checkpoints: ASethi04/pi05-BimanualYAM-oranges (realsense), ultrawide variants pending.

## Server (remote_inference) — apply pi05_serving.patch first
Without it: (1) generic branch loads weights from the checkpoint config's own
`pretrained_path` → base weights or dead path, silently (the published repo ships
`pretrained_path: null` so a stock server fails loudly instead); (2) warmup crashes —
manifest state_dim=32 padded vs (14,) normalizer stats; (3) handshake rejects the YAM
client — 32 synthetic state names + renamed camera slots vs 14 real names + top/left/right.

## Client (lerobot-rollout), local/sync mode
- REQUIRED: `--rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.left":"observation.images.left_wrist_0_rgb","observation.images.right":"observation.images.right_wrist_0_rgb"}'`
  (skips the visual-mismatch fail-fast AND re-installs the map that an empty CLI
  rename_map would otherwise wipe from the checkpoint's preprocessor — context.py:485-489).
  For the no-top ultrawide checkpoint drop the "top" entry.
- Cameras must be named exactly top/left/right in --robot.cameras (keys used verbatim).
- `--robot.max_gripper_delta=0.05` (default 0.03 throttles grasps; data has 0.05/tick).
- Remote mode needs no rename flags (manifest-driven) once the server patch is applied.
- fp32 inference ≈16GB weights: use a ≥24GB GPU. bf16 autocast at inference is safe.
- Task string exactly: `Put all oranges in the bowl`
