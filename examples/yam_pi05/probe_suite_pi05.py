#!/usr/bin/env python
"""Offline probe suite for pi05 YAM checkpoints — five probes beyond L1 gates.

  A baseline     stride-sampled chunk L1 (reference for all deltas)
  B grasp_events event-level gripper analysis: at each ground-truth grasp/release
                 (hysteresis 0.78/0.82 from the data audit), does a chunk starting
                 k ticks earlier contain the closure, and with what timing error?
  C stitching    chunks predicted at t and t+15: L1 over the 15 overlapping steps
                 (replan stability; informs execution_horizon; memorization-free)
  D counterfact  same stride sample under: hue-rotated frames (orange->green),
                 brightness control, black frames, frozen-mean state, noise state
  E task_string  alternate + nonsense prompt sensitivity

One-sided caveats apply: these detect bad models; they cannot certify good ones.
"""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TF

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

FPS, CHUNK, ADIM = 30, 30, 14
GRIP_IDX = [6, 13]
JOINT_IDX = [i for i in range(ADIM) if i not in GRIP_IDX]
CLOSE_T, OPEN_T = 0.78, 0.82  # audit-derived gripper hysteresis


def load_policy(ckpt, device="cuda"):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    # MolmoAct2 checkpoints from runs 2/3 saved inference_action_mode=None
    if hasattr(cfg, "inference_action_mode") and cfg.inference_action_mode is None:
        cfg.inference_action_mode = "continuous"
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt))
    return cfg, policy, pre, post


class Runner:
    def __init__(self, ckpt, root, repo_id, episodes, device="cuda"):
        self.device = device
        self.cfg, self.policy, self.pre, self.post = load_policy(ckpt, device)
        # MolmoAct2 runs bf16 under autocast (matches its published eval); pi05 is fp32
        self.bf16 = next(self.policy.parameters()).dtype == torch.bfloat16
        self.ds = LeRobotDataset(
            repo_id=repo_id, root=root, episodes=episodes,
            delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
            image_transforms=None, video_backend="pyav",
        )
        self.episodes = episodes
        # subset-local [from, to) per episode: absolute meta indices mapped through
        # absolute_to_relative_idx (episode_data_index does not exist in this version;
        # filtered rows come back in storage order, not request order)
        a2r = self.ds.absolute_to_relative_idx
        self.ep_range = {}
        for ep in episodes:
            em = self.ds.meta.episodes[ep]
            lo_abs, hi_abs = int(em["dataset_from_index"]), int(em["dataset_to_index"])
            self.ep_range[ep] = (a2r[lo_abs], a2r[hi_abs - 1] + 1)
        # full-resolution per-episode action arrays straight from parquet
        self.ep_actions = {}
        root = Path(root)
        for f in sorted(root.glob("data/**/*.parquet")):
            t = pq.read_table(f, columns=["action", "episode_index"])
            acts = np.stack(t.column("action").to_numpy(zero_copy_only=False))
            eps = t.column("episode_index").to_numpy()
            for ep in np.unique(eps):
                if int(ep) in episodes:
                    self.ep_actions.setdefault(int(ep), []).append(acts[eps == ep])
        self.ep_actions = {k: np.concatenate(v) for k, v in self.ep_actions.items()}
        for ep, (lo, hi) in self.ep_range.items():
            assert hi - lo == len(self.ep_actions[ep]), f"range/parquet mismatch ep {ep}"

    def collate(self, idxs):
        items = [self.ds[i] for i in idxs]
        batch = {}
        for k, v in items[0].items():
            if torch.is_tensor(v):
                batch[k] = torch.stack([it[k] for it in items])
            else:
                batch[k] = [it[k] for it in items]
        return batch

    @torch.inference_mode()
    def predict(self, batch, mutate=None):
        # fixed seed per call: flow-matching noise draws cancel exactly across
        # conditions, so mean_abs_pred_delta==0 for a truly ignored mutation
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        batch = {k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in batch.items()}
        if mutate is not None:
            batch = mutate(batch)
        batch["robot_type"] = "bi_yam_follower"  # MolmoAct2 reads it; pi05 ignores it
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else contextlib.nullcontext()
        with ctx:
            proc = self.pre(batch)
            chunk = self.policy.predict_action_chunk(proc)
            out = torch.stack([self.post(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)
        self.policy.reset()
        if hasattr(self.pre, "reset"):
            self.pre.reset()
        if hasattr(self.post, "reset"):
            self.post.reset()
        return out.float().cpu().numpy()

    def run_condition(self, idxs, mutate=None, bs=4):
        preds, gts, pads = [], [], []
        for s in range(0, len(idxs), bs):
            batch = self.collate(idxs[s:s + bs])
            gts.append(batch["action"].numpy().copy())
            pad = batch.get("action_is_pad")
            pads.append(pad.numpy().copy() if pad is not None
                        else np.zeros(gts[-1].shape[:2], bool))
            preds.append(self.predict(batch, mutate))
        P, G, M = np.concatenate(preds), np.concatenate(gts), np.concatenate(pads)
        m = (~M)[..., None]
        l1 = (np.abs(P - G) * m).sum(0) / np.maximum(np.broadcast_to(m, G.shape).sum(0), 1)
        return P, {"l1_joints": float(l1[:, JOINT_IDX].mean()),
                   "l1_grippers": float(l1[:, GRIP_IDX].mean())}


def image_keys(batch):
    return [k for k in batch if str(k).startswith("observation.images.")]


def mut_hue(batch):
    for k in image_keys(batch):
        batch[k] = torch.stack([TF.adjust_hue(img, 0.35) for img in batch[k]])
    return batch


def mut_brightness(batch):
    for k in image_keys(batch):
        batch[k] = torch.stack([TF.adjust_brightness(img, 0.75) for img in batch[k]])
    return batch


def mut_black(batch):
    for k in image_keys(batch):
        batch[k] = torch.zeros_like(batch[k])
    return batch


def mut_shuffle(batch):
    for k in image_keys(batch):
        batch[k] = batch[k].roll(1, dims=0)
    return batch


def make_mut_state_const(vec):
    t = torch.as_tensor(vec, dtype=torch.float32)

    def mut(batch):
        batch["observation.state"] = t.repeat(batch["observation.state"].shape[0], 1)
        return batch
    return mut


def make_mut_state_noise(q01, q99, seed=0):
    rng = np.random.default_rng(seed)

    def mut(batch):
        n = batch["observation.state"].shape[0]
        s = rng.uniform(q01, q99, size=(n, len(q01))).astype(np.float32)
        batch["observation.state"] = torch.from_numpy(s)
        return batch
    return mut


def make_mut_task(text):
    def mut(batch):
        batch["task"] = [text] * len(batch["task"])
        return batch
    return mut


def grasp_events(g):
    events, state = [], ("open" if g[0] > 0.8 else "closed")
    for t in range(1, len(g)):
        if state == "open" and g[t] < CLOSE_T:
            events.append((t, "grasp"))
            state = "closed"
        elif state == "closed" and g[t] > OPEN_T:
            events.append((t, "release"))
            state = "open"
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    r = Runner(a.ckpt, a.root, a.repo_id, a.episodes)
    stats = r.ds.meta.stats["observation.state"]
    results = {"ckpt": a.ckpt, "episodes": a.episodes}

    # ---- A: baseline on the stride sample ----
    # trim to a multiple of the batch size: a singleton batch makes roll(1) a no-op
    # in the shuffle condition (the historical false-vision-blindness trap)
    stride_idxs = list(range(0, len(r.ds), a.stride))
    stride_idxs = stride_idxs[: len(stride_idxs) - (len(stride_idxs) % 4)] or stride_idxs
    base_P, base = r.run_condition(stride_idxs)
    results["baseline"] = base
    print("baseline", base, flush=True)

    # ---- D/E: counterfactual conditions on the same sample ----
    q01 = np.asarray(stats["q01"], dtype=np.float64).reshape(-1)
    q99 = np.asarray(stats["q99"], dtype=np.float64).reshape(-1)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    conditions = {
        "identity_rerun_control": None,  # noise-floor check: must be ~0 after seeding
        "hue_rotated": mut_hue,
        "brightness_control": mut_brightness,
        "images_black": mut_black,
        "images_shuffled": mut_shuffle,
        "state_frozen_mean": make_mut_state_const(mean),
        "state_noise": make_mut_state_noise(q01, q99),
        "task_alternate": make_mut_task("Put all apples in the basket"),
        "task_nonsense": make_mut_task("asdf qwerty zxcv"),
    }
    for name, mut in conditions.items():
        P, m = r.run_condition(stride_idxs, mutate=mut)
        m["delta_joints_vs_baseline"] = round(m["l1_joints"] - base["l1_joints"], 5)
        m["ratio_joints_vs_baseline"] = round(m["l1_joints"] / max(base["l1_joints"], 1e-9), 2)
        m["mean_abs_pred_delta"] = float(np.abs(P - base_P).mean())
        results[name] = m
        print(name, m, flush=True)

    # ---- B: grasp-event timing ----
    ev_records = []
    for ep in a.episodes:
        acts = r.ep_actions[ep]
        lo, hi = r.ep_range[ep]
        eplen = hi - lo
        for gi, g in enumerate(GRIP_IDX):
            for t_ev, kind in grasp_events(acts[:, g]):
                for k in (2, 8, 14, 20, 26):
                    t0 = t_ev - k
                    if not (0 <= t0 < eplen):
                        continue
                    # clean pre-event window only: a quick re-grasp inside the
                    # window would let the chunk trivially 'contain' the event
                    pre_win = acts[t0:t_ev, g]
                    if kind == "grasp" and (len(pre_win) == 0 or pre_win.min() <= CLOSE_T):
                        continue
                    if kind == "release" and (len(pre_win) == 0 or pre_win.max() >= OPEN_T):
                        continue
                    ev_records.append({"ep": ep, "grip": gi, "kind": kind,
                                       "t0_local": lo + t0, "offset": k})
    det, timing_errs, miss = {"grasp": 0, "release": 0}, [], {"grasp": 0, "release": 0}
    n_by_kind = {"grasp": 0, "release": 0}
    for s in range(0, len(ev_records), 4):
        chunk_recs = ev_records[s:s + 4]
        idxs = [rec["t0_local"] for rec in chunk_recs]
        batch = r.collate(idxs)
        P = r.predict(batch)
        for bi, rec in enumerate(chunk_recs):
            g = GRIP_IDX[rec["grip"]]
            pg = P[bi, :, g]
            if rec["kind"] == "grasp":
                hits = np.where(pg < CLOSE_T)[0]
            else:
                hits = np.where(pg > OPEN_T)[0]
            n_by_kind[rec["kind"]] += 1
            if len(hits):
                det[rec["kind"]] += 1
                timing_errs.append(int(hits[0]) - rec["offset"])
            else:
                miss[rec["kind"]] += 1
    timing_errs = np.asarray(timing_errs, dtype=np.float64)
    results["grasp_events"] = {
        "n_events_scored": len(ev_records),
        "n_by_kind": n_by_kind,
        "detection_rate": {k: (det[k] / n_by_kind[k] if n_by_kind[k] else None)
                           for k in det},
        "timing_err_mean_ticks": float(timing_errs.mean()) if len(timing_errs) else None,
        "timing_err_abs_mean_ticks": float(np.abs(timing_errs).mean()) if len(timing_errs) else None,
        "timing_err_p90_abs": float(np.quantile(np.abs(timing_errs), 0.9)) if len(timing_errs) else None,
    }
    print("grasp_events", results["grasp_events"], flush=True)

    # ---- C: chunk-stitching consistency ----
    pairs = []
    for ep in a.episodes:
        lo, hi = r.ep_range[ep]
        for t in range(lo, hi - CHUNK - 15, a.stride):
            pairs.append((t, t + 15))
    overlap_l1 = np.zeros(15)
    n_pairs = 0
    for s in range(0, len(pairs), 2):
        chunk_pairs = pairs[s:s + 2]
        idxs = [i for p in chunk_pairs for i in p]
        batch = r.collate(idxs)
        P = r.predict(batch)
        for pi in range(len(chunk_pairs)):
            a1, a2 = P[2 * pi], P[2 * pi + 1]
            overlap_l1 += np.abs(a1[15:30] - a2[0:15]).mean(axis=1)
            n_pairs += 1
    overlap_l1 /= max(n_pairs, 1)
    results["stitching"] = {
        "n_pairs": n_pairs,
        "overlap_l1_mean": float(overlap_l1.mean()),
        "overlap_l1_by_position": overlap_l1.round(5).tolist(),
    }
    print("stitching", results["stitching"], flush=True)

    a.out.write_text(json.dumps(results, indent=2))
    print("PROBE_SUITE_DONE")


if __name__ == "__main__":
    main()
