# Deploying a UMI end-effector-pose policy on a bimanual YAM

This is the deployment guide for policies whose action space is per-arm **absolute
end-effector pose** (xyz + rotation vector in an episode-start frame, plus a gripper
width) — the shape you get from fine-tuning on handheld UMI-gripper demonstrations. The
target is `BiYAM`, which speaks **joint space**.

Two modules do the work:

| module | role |
|---|---|
| `umi_ee_client.py` | the wire: owns the embodiment manifest the stock rollout path cannot express, talks gRPC to `lerobot-policy-server` |
| `yam_umi_ee_bridge.py` | the conversion: FK/IK, frames, gripper mapping, safety caps, plus offline test and calibration tools |

## Read this before you plan a rollout

**A policy trained on handheld-gripper data is not automatically executable on this arm.**
Measured on our dual-UMI checkpoint against real YAM start configurations, on held-out
episodes: **19.6%** of the policy's target poses are reachable with no frame calibration,
and **52.0%** after calibrating the tool frame. So roughly half the commanded poses are
outside the arm's reachable set even after calibration.

This is a known property of the approach, not a bug in this code. UMI's own follow-up
measured *"a significant portion (32.5%) of trials ... fail during placement due to
violations of the robot's inverse kinematics constraints"*, and UMI's limitations section
names it as open work: *"Future works could develop an embodiment-aware policy learning
framework that can transfer skills from valid but hardware-infeasible actions."* Other
groups report the same magnitude — FeasibleCap measured 83% infeasible frames in unguided
handheld collection; ARMADA measured 1.3% replay success without feasibility feedback.

The tooling here lets you **measure this for your own rig and checkpoint before you power
the arms**, and gives you the two mitigations that are safe to apply at deployment. It does
not pretend to close the gap. See "If the numbers are bad" at the end.

## 1. Serve the policy

```bash
export HF_HOME=/path/to/hf-cache      # base MolmoAct2 ckpt + FAST tokenizer must be cached
export HF_HUB_OFFLINE=1
lerobot-policy-server \
  --policy.pretrained_name_or_path=/path/to/checkpoints/012000/pretrained_model \
  --policy.device=cuda --host=0.0.0.0 --port=8081
```

Pass **nothing else**. `--policy.type` / `--policy.norm_tag` divert into the
original-checkpoint branch, which reloads the *base* model's normalization statistics and
would silently serve wrong actions; dtype and `inference_action_mode` already come from the
checkpoint config. Needs ~48 GB host RAM and a GPU newer than Volta (a bf16 checkpoint will
not run on a V100 — `no kernel image is available for execution on the device`).

## 2. Check the math, with no robot and no server

```bash
python -m lerobot.remote_inference.yam_umi_ee_bridge --self-test \
    --urdf /path/to/i2rt/robot_models/arm/yam/yam.urdf \
    --yam_dataset /path/to/yam-teleop-dataset --repo_id user/yam-teleop --episode 3
```

Verifies the frame algebra, the gripper map, the safety caps, and — on real recorded YAM
trajectories — that FK→IK round-trips, that the adapter is the identity when the policy
echoes its own state, and that unreachable targets raise instead of returning a nearest
pose. Expect `SELF_TEST_OK`.

## 3. Measure the embodiment gap for *your* rig

```bash
python -m lerobot.remote_inference.yam_umi_ee_bridge --calibrate \
    --urdf .../yam.urdf --yam_dataset ... --repo_id ... --umi_dataset ... \
    --episodes 3 17 41 --holdout_episodes 62 88
```

Searches tool-frame rotations and reports the reachable fraction on a **held-out** set.
Believe the held-out column: a rotation fitted to a single episode overfits so badly that it
scored *worse than no calibration* in our tests, while a three-episode fit generalized
(19.6% → 52.0%).

A large gain here is expected and matches published practice: UMI hit the same mismatch and
solved it in hardware, 3D-printing an adapter that *"rotates WSG50 gripper 90-degree with
respect to the robot's end-effector flange"* because their gripper is held horizontally
while the arm is built for top-down picking. Prefer measuring your physical mount over
trusting the fitted number; use the fit as a cross-check.

## 4. Dry-run the whole loop against a live server

```bash
python -m lerobot.remote_inference.yam_umi_ee_bridge --replay \
    --server host:8081 --urdf ... --yam_dataset ... --repo_id ... \
    --umi_dataset ... --umi_repo_id ... --task "Put all oranges in the bowl" \
    --tool_offset RX RY RZ --tracking best_effort --roll_tolerance 3.14159
```

Runs the real client and real kinematics with a recorded episode standing in for the arm,
and reports executed commands, max joint step, bounds, and IK failures.

## 5. On the rig

```python
adapter = YamUmiEeAdapter(
    left=YamArmKinematics(urdf), right=YamArmKinematics(urdf),
    frames=EpisodeFrames(tool_offset=my_measured_mount),
    tracking=TrackingPolicy(mode="best_effort", roll_tolerance=np.pi),
)
client = UmiEeRemoteClient(UmiEeClientConfig(
    server_address="gpu-host:8081", task="Put all oranges in the bowl",
    camera_keys=("umi1", "umi2"), image_width=800, image_height=600, control_hz=30.0))
client.connect()

# once per episode, at the start pose
client.reset()
adapter.capture_episode_start(robot.get_observation())

# each tick
observation = robot.get_observation()
chunk = client.predict(adapter.observation_to_policy_state(observation), images, tick)
for row in chunk[:15]:                       # execution horizon, then re-query
    observation = robot.get_observation()
    robot.send_action(adapter.action_row_to_joint_command(row, observation))
    if max(e[0] for e in adapter.last_residual.values()) > 0.01:
        ...  # tracking error over 10 mm: the arm is not following the policy
```

Run `BiYAM` with **`--robot.max_gripper_delta=0.05`**. The default 0.03 throttles ~3.4% of
ticks, 92–98% of them within half a second of a grasp or release, stretching a close from
0.33 s to 0.55 s.

## The three conversions that are silent when wrong

**Units.** `RobotKinematics` takes and returns **degrees**; YAM joints are **radians**. The
dangerous direction is feeding radians as degrees: the arm still sits a plausible ~0.21 m
from its base at every frame and looks fine — only the workspace *spread* collapses.
`YamArmKinematics` owns the conversion and asserts on magnitude.

**Gripper.** The policy emits UMI jaw width / 125 mm; YAM wants a normalized `gripper.pos`.
Both are "larger = more open", so nothing looks wrong — but commanding the policy value
directly leaves the jaws **11–15 mm wider than the orange at every grasp**, traveling only
47% of the required distance. No contact at all, with a pose trajectory that looks perfect
on video. `GripperMap` fixes the scale, per arm (the two UMI devices differ: 123.5 mm vs
118.1 mm maximum aperture, while agreeing on the grasped orange to 0.08 mm).

**Frames.** Policy poses are relative to each arm's episode-start pose, so the anchor must
be re-captured on every `client.reset()`. A tool-frame offset does **not** cancel out: it
conjugates the relative motion (`T0 · C · P · C⁻¹`), which is why calibrating it changes
reachability so much.

## Two mitigations, and what they cost

**`roll_tolerance`** — a parallel jaw is symmetric about its approach axis, so gripper roll
is a *ranged* goal, not a fixed one. Both YAM arms are 6-DoF driving a 6-DoF task, so
`(I − J⁺J) = 0` and every classical joint-limit-avoidance method is mathematically inert;
declaring roll a tolerance band manufactures the missing degree of freedom for free
(RangedIK, ICRA 2023). This is **not** a distortion of the task — it picks a physically
equivalent grasp. Measured: median residual 43.7 mm → 9.6 mm.

**`tracking="best_effort"`** — tracks unreachable targets position-first instead of raising,
mirroring the Cartesian impedance controllers used in UMI deployments (their Franka holds
orientation ~50× more softly than position). This made our replay run clean: 180/180
commands, 0 IK failures.

> **Do not read "it runs" as "it works."** Naive projection of infeasible commands is
> measured to raise IK feasibility to 99.9% while collapsing task success from 75.0% to
> **23.7%** (arXiv 2606.24208). `best_effort` converts hard failures into silent tracking
> error. Use `strict` during bring-up so mismatches are impossible to miss, watch
> `adapter.last_residual` always, and treat a large residual as a failed rollout rather than
> a completed one.

## If the numbers are bad

Ranked by evidence, from the literature survey in `notes_retargeting_literature.md` and
`notes_umi_literature.md`:

1. **Fine-tune on a small amount of on-robot data.** Six independent replications; EgoVLA
   reports **0% zero-shot** without any robot data. 50–150 teleop demos at roughly 10:1
   human:robot is the usual recipe. Strongest evidence, moderate cost.
2. **Feasibility-aware data curation, then retrain.** VISTA measured, at a fixed 50 demos,
   **OSR 0.00 vs 0.65** between low- and high-feasibility subsets — and the same low subset
   scored 0.80 on a *different* robot, so the filter must be embodiment-conditioned. Prefer
   re-weighting to hard filtering: hard filtering cost H2O 35% of its data for +4.6 points,
   and here the discarded data would be wrist-rotation-correlated, deleting a behavior mode
   rather than noise.
3. **Re-anchor actions to the current EE pose and retrain.** UMI's ablation is stark —
   relative 20/20, delta 16/20, absolute 5/20 — and their code anchors to the last
   *measured* pose (`env_obs['robot0_eef_pos'][-1]`), re-anchored every chunk, not to the
   episode start. Our data uses the episode-start convention, i.e. their absolute baseline.
   This is a relabelling of existing data, no re-collection. Caveat: it changes the failure
   *mode* (hard IK error → graceful lag) more than it creates workspace, and FeasibleCap saw
   83% infeasibility under exactly this convention — so treat it as a cheap A/B, not a
   guaranteed win.
4. **Optimize the episode start pose and the arm's base placement.** Our reachability
   numbers are measured *from one start configuration*, which is a free parameter nobody has
   tuned. Reachability-aware placement is cheap and setup-only.
5. **Feasibility guidance inside the sampling loop.** MolmoAct2's flow-matching action
   expert has exactly the iterative sampler that EADP (UMI-on-Air) and constrained flow
   matching hook into, deployment-only and no retraining; published gains of +9%/+20% and
   68.1% → 81.6% task success. Novel work for this stack, but the hook exists.

**One measurement worth writing up.** Across the ~20 papers surveyed, nobody reports the
human-versus-robot wrist mobility asymmetry. In our corpora the handheld device rotates a
median **31°** per episode while the YAM flange in its own teleop demos rotates **0.6°** —
a 50× difference in a quantity that is invariant to every frame convention, and therefore
something no client-side calibration can fix.
