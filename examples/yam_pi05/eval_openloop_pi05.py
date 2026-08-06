#!/usr/bin/env python
"""Open-loop 30-step action-chunk error for pi05 YAM checkpoints.

Adapted from eval_openloop.py (MolmoAct2). Differences:
  - no inference_action_mode (MolmoAct2-only field)
  - image keys for the camera-shuffle probe are derived from the batch
    (the checkpoint's saved preprocessor renames top/left/right ->
    base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb internally)
  - fp32 end-to-end (matches training numerics; no autocast)

HONEST CAVEAT: training used a 100/0 split, so every frame scored below was
seen during training. These numbers are a REGRESSION GATE and a MEMORIZATION
probe -- they are NOT a checkpoint selector.

Comparison baselines (MolmoAct2 run 2, same protocol, episodes [3,17,41,62,78],
stride 60): joint L1 0.0515 rad, gripper L1 0.0036, shuffled-cam ratio 1.53x,
jerk 0.002.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

FPS, CHUNK, ADIM = 30, 30, 14
JOINT_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP_IDX = [6, 13]
FEATS = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_gripper",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    "right_joint_5", "right_gripper",
]
# YAM command-space joint limits (simulators/bi_yam/config.py), per arm
JOINT_LIMITS = [
    (-2.618, 3.142), (0.0, 3.665), (0.0, 3.142),
    (-1.693, 1.571), (-1.571, 1.571), (-2.094, 2.094),
]


def load_policy(ckpt, device="cuda"):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(device).eval()
    # pretrained_path restores THIS checkpoint's own saved processor pipelines:
    # rename map + corrected quantile stats travel with the checkpoint.
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt))
    return cfg, policy, pre, post


def make_loader(root, repo_id, episodes, stride, batch_size, num_workers):
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
        image_transforms=None,  # eval must NOT augment
        video_backend="pyav",
    )
    idx = list(range(0, len(ds), stride))
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
    )
    return ds, loader


def limit_violations(pred):
    """Count predictions outside YAM joint limits / gripper [0,1]."""
    viol = 0
    for arm in (0, 7):
        for j, (lo, hi) in enumerate(JOINT_LIMITS):
            v = pred[..., arm + j]
            viol += int(((v < lo) | (v > hi)).sum())
    for g in GRIP_IDX:
        v = pred[..., g]
        viol += int(((v < 0.0) | (v > 1.0)).sum())
    return viol


@torch.inference_mode()
def eval_ckpt(ckpt, loader, device="cuda", shuffle_images=False):
    cfg, policy, pre, post = load_policy(ckpt, device)
    abs_err = np.zeros((CHUNK, ADIM))
    n = np.zeros((CHUNK, ADIM))
    preds = []

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        gt = batch["action"].float().cpu().numpy()
        pad = batch.get("action_is_pad")
        pad = pad.cpu().numpy() if pad is not None else np.zeros(gt.shape[:2], bool)

        if shuffle_images:  # vision-sensitivity probe (raw dataset keys, pre-rename)
            for k in list(batch.keys()):
                if str(k).startswith("observation.images."):
                    batch[k] = batch[k].roll(1, dims=0)

        proc = pre(batch)
        chunk = policy.predict_action_chunk(proc)
        # postprocessor applied PER-TIMESTEP (mirrors remote_inference/backend.py)
        out = torch.stack([post(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)

        pred = out.float().cpu().numpy()
        preds.append(pred)
        m = (~pad)[..., None]
        abs_err += (np.abs(pred - gt) * m).sum(0)
        n += np.broadcast_to(m, gt.shape).sum(0)
        policy.reset()
        if hasattr(pre, "reset"):
            pre.reset()
        if hasattr(post, "reset"):
            post.reset()

    l1 = abs_err / np.maximum(n, 1)
    P = np.concatenate(preds, 0)
    return {
        "n_samples": int(P.shape[0]),
        "l1_per_dim": dict(zip(FEATS, l1.mean(0).round(5).tolist())),
        "l1_joints": float(l1[:, JOINT_IDX].mean()),
        "l1_grippers": float(l1[:, GRIP_IDX].mean()),
        "l1_by_horizon": l1.mean(1).round(5).tolist(),
        "l1_step0": float(l1[0].mean()),
        "l1_step29": float(l1[-1].mean()),
        # degeneracy / safety probes
        "pred_std_across_samples": float(P.std(0).mean()),
        "pred_std_within_chunk": float(P.std(1).mean()),
        "gripper_std": float(P[..., GRIP_IDX].std()),
        "pred_min": P.min(0).min(0).round(4).tolist(),
        "pred_max": P.max(0).max(0).round(4).tolist(),
        "limit_violations": limit_violations(P),
        "jerk_mean": float(np.abs(np.diff(P, n=2, axis=1)).mean()),
        "nonfinite": int((~np.isfinite(P)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True, help=".../checkpoints")
    ap.add_argument("--root",
                    default="/n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/data/yam-vive-teleop-fixedstats")
    ap.add_argument("--repo_id", default="brandonyang/yam-vive-teleop")
    ap.add_argument("--episodes", type=int, nargs="+", default=[3, 17, 41, 62, 78])
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--camera_probe", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("openloop_eval_pi05.json"))
    a = ap.parse_args()

    _, loader = make_loader(a.root, a.repo_id, a.episodes, a.stride, a.batch_size, a.num_workers)
    results = {}
    for step_dir in sorted(a.run_dir.glob("0*")):
        ck = step_dir / "pretrained_model"
        if not (ck / "model.safetensors").exists():
            continue
        results[step_dir.name] = eval_ckpt(ck, loader)
        if a.camera_probe:
            results[step_dir.name]["shuffled_cams"] = eval_ckpt(ck, loader, shuffle_images=True)
        r = results[step_dir.name]
        print(f"{step_dir.name}: l1_joints={r['l1_joints']:.4f} l1_grip={r['l1_grippers']:.4f} "
              f"viol={r['limit_violations']} nonfinite={r['nonfinite']} "
              f"pred_std={r['pred_std_across_samples']:.4f}", flush=True)
        a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
