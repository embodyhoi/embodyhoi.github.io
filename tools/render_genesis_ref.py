#!/usr/bin/env python
"""Render kinematic replays of retargeted G1 + XHand references (ARCTIC objects) in Genesis, headless.

The real training env (ArcticTestEnv, same Hydra overrides as outputs_icra/train_one.sh) is instantiated with
num_envs=1; then, instead of policy actions, each frame directly sets the robot DOFs from the reference
(robot.residual_qpos) and the object root pose + articulation from the demo (object.demo_states).

Example:
  CUDA_VISIBLE_DEVICES=0 python render_genesis_ref.py --sid gregen_00025 --view both --frame-dump
  python render_genesis_ref.py --sid gregen_00025 --rollout /path/eval_ep0.npy   # replay a policy rollout instead
"""
import os
import re
import sys
import time
import argparse

ENV_DEFAULTS = dict(
    PYTHONNOUSERSITE="1", SETUPTOOLS_USE_DISTUTILS="stdlib", PYOPENGL_PLATFORM="egl",
    HYDRA_FULL_ERROR="1", PYTHONBREAKPOINT="0", WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="0",
    # EGL device for the offscreen rasterizer (index into eglQueryDevicesEXT == nvidia-smi index on this box).
    # GPU 0's GL context renders black at ~9 s/frame here, so the (tiny) GL context goes on GPU 6; physics stays on GPU 0.
    EGL_DEVICE_ID="6",
)
for k, v in ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

ROOT = "/scr/kartiksh/dexmachina"
PKG = os.path.join(ROOT, "dexmachina")          # package dir; relative assets/ paths are used there
CONFIGS = os.path.join(PKG, "configs")
RETARGET_LOGS = os.path.join(ROOT, "outputs_icra", "retarget_logs")
DEFAULT_OUT = "/scr/kartiksh/embodyhoi-website/media/genesis"

os.chdir(PKG)
sys.path.insert(0, ROOT)

import numpy as np
import torch
import imageio
import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")

from hydra import initialize_config_dir, compose  # noqa: E402
import dexmachina.envs.arctic_test as arctic_test  # noqa: E402

# camera framings: 3/4 front view (robot faces -y, stands at y~0.23; object on the table at y~-0.25, z~1.0)
# camera on the robot's right (-x) side: the open lid of the box leans away from it instead of occluding the hands
VIEWS = {
    "wide": dict(res=(1920, 1080), pos=(-1.15, -1.95, 1.55), lookat=(0.02, -0.02, 1.12), fov=24),
    "square": dict(res=(1080, 1080), pos=(-0.95, -1.60, 1.45), lookat=(0.0, -0.02, 1.08), fov=28),
}
# low-res candidates for camera exploration (--view probe renders only first/mid/last PNGs of each)
PROBE_VIEWS = {
    "pA_left34": dict(res=(960, 540), pos=(1.2, -1.9, 1.55), lookat=(0.02, -0.02, 1.15), fov=24),
    "pB_right34": dict(res=(960, 540), pos=(-1.2, -1.9, 1.55), lookat=(0.02, -0.02, 1.15), fov=24),
    "pC_front": dict(res=(960, 540), pos=(0.0, -2.3, 1.5), lookat=(0.02, -0.02, 1.15), fov=24),
    "pD_left_mild": dict(res=(960, 540), pos=(0.8, -2.1, 1.5), lookat=(0.02, -0.02, 1.15), fov=24),
    "pE_right_mild": dict(res=(960, 540), pos=(-0.8, -2.1, 1.5), lookat=(0.02, -0.02, 1.15), fov=24),
    "pF_left_sq": dict(res=(540, 540), pos=(0.9, -1.6, 1.45), lookat=(0.03, -0.1, 1.08), fov=26),
    "pG_right_sq": dict(res=(540, 540), pos=(-0.9, -1.6, 1.45), lookat=(0.03, -0.1, 1.08), fov=26),
}


def parse_done_line(sid):
    """Mirror train_one.sh: parse the DONE line of retarget_logs/<sid>.out."""
    path = os.path.join(RETARGET_LOGS, f"{sid}.out")
    done = [l for l in open(path) if "DONE base_name" in l]
    assert done, f"no DONE line in {path}"
    line = done[-1].strip()
    g = lambda key: re.search(rf"{key}=([^ ]+)", line).group(1)
    meta = dict(base_name=g("base_name"), ref=g("ref"), episode_length=int(g("episode_length")), base=g("base"))
    meta["obj"] = re.sub(r"^[a-z]+_[0-9]+_([a-z]+)_0$", r"\1", meta["base_name"])
    meta["cond"] = {"gregen": "guided", "demo": "unguided"}.get(sid.split("_")[0], sid.split("_")[0])
    return meta


def hydra_overrides(meta):
    obj, bn = meta["obj"], meta["base_name"]
    return [
        "task=g1_arctic_xhand_sampled",
        f"task.obj_cfg.name={obj}", f"task.obj_cfg.load_kwargs.file={obj}/decomp/{obj}_decomp.urdf",
        f"task.task_cfg.clip={obj}-30-230",
        f"task.robot_cfg.load_kwargs.base_name={bn}", f"task.robot_cfg.load_kwargs.ref_qpos_file={meta['ref']}",
        f"task.robot_cfg.load_kwargs.pos=[{meta['base']}]",
        "task.task_cfg.skip_initial_frames=0", "task.task_cfg.skip_final_frames=8",
        f"task.env_cfg.episode_length={meta['episode_length']}",
        "task.rew_cfg.use_contact_reward=true", "task.rew_cfg.use_retarget_contact=true",
        "task.task_cfg.observe_contact_force=true", "task.robot_cfg.action_cfg.use_finger=true",
        "task.obj_cfg.actuators.kp=0.0", "task.obj_cfg.actuators.kv=0.0",
    ]


def build_env(meta, views, extra_overrides=()):
    """Instantiate ArcticTestEnv (num_envs=1) with our cameras + white studio lighting patched in at runtime."""
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="train_grasp", overrides=hydra_overrides(meta) + list(extra_overrides))
    cfg.task.env_cfg.num_envs = 1
    cfg.task.env_cfg.record_video = False
    cfg.task.env_cfg.is_eval = True
    cfg.task.robot_cfg.action_cfg.arm_control_mode = "joint"

    orig_scene_cfg = arctic_test.get_grasp_scene_cfg
    orig_build = gs.Scene.build
    cams = {}

    def patched_scene_cfg(*a, **k):
        scene_cfg = orig_scene_cfg(*a, **k)
        vo = scene_cfg["vis_options"].model_dump()
        vo.update(
            background_color=(1.0, 1.0, 1.0),
            ambient_light=(0.45, 0.45, 0.45),
            shadow=True,
            lights=[
                {"type": "directional", "dir": (-0.6, 0.8, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 3.0},
                {"type": "directional", "dir": (0.8, -0.3, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
            ],
        )
        scene_cfg["vis_options"] = gs.options.VisOptions(**vo)
        return scene_cfg

    def patched_build(self, *a, **k):
        for name, v in views.items():
            cams[name] = self.add_camera(res=tuple(v["res"]), pos=v["pos"], lookat=v["lookat"], fov=v["fov"], GUI=False)
        return orig_build(self, *a, **k)

    arctic_test.get_grasp_scene_cfg = patched_scene_cfg
    gs.Scene.build = patched_build
    try:
        env = arctic_test.ArcticTestEnv(
            env_cfg=cfg.task.env_cfg, robot_cfg=cfg.task.robot_cfg, obj_cfg=cfg.task.obj_cfg,
            task_cfg=cfg.task.task_cfg, rew_cfg=cfg.task.rew_cfg, rl_cfg=cfg.rl.config,
            device=torch.device("cuda:0"),
        )
    finally:
        arctic_test.get_grasp_scene_cfg = orig_scene_cfg
        gs.Scene.build = orig_build
    return env, cams


def ground_seg_id(scene, ground):
    """Seg id of the ground plane for segmentation_level='entity' (0 is background, so usually entity idx + 1)."""
    d = getattr(scene, "segmentation_idx_dict", None)
    if d:
        for seg, val in d.items():
            if val == ground.idx or (isinstance(val, (tuple, list)) and val and val[0] == ground.idx):
                return seg
    return ground.idx + 1


def load_states(env, rollout=None):
    """Returns (qpos (T,53) float64, obj (T,8)) numpy arrays: reference by default, or a policy rollout dump."""
    robot, obj = env.robot, env.object
    if rollout is None:
        q = robot.residual_qpos.detach().cpu().numpy().astype(np.float64)
        o = obj.demo_states.detach().cpu().numpy().astype(np.float64)
        T = min(len(q), len(o))
        return q[:T], o[:T]
    d = np.load(rollout, allow_pickle=True).item()
    T = len(d["arm_states"])
    q = np.repeat(robot.init_qpos[0].detach().cpu().numpy().astype(np.float64)[None], T, 0)
    q[:, robot.control_upper_idxs] = d["arm_states"]
    q[:, robot.control_finger_idxs] = d["finger_states"]
    arti = d["object_arti"].reshape(T, -1)
    o = np.concatenate([d["object_pos"], d["object_quat_wxyz"], arti], axis=1).astype(np.float64)
    return q, o


def interpolate(q, o, factor):
    """Linear in qpos / pos / arti, normalized-lerp (shortest arc) for the quaternion."""
    if factor <= 1:
        return q, o
    T = len(q)
    t_src = np.arange(T)
    t_new = np.linspace(0, T - 1, (T - 1) * factor + 1)
    qi = np.stack([np.interp(t_new, t_src, q[:, j]) for j in range(q.shape[1])], 1)
    quat = o[:, 3:7].copy()
    for i in range(1, T):  # make consecutive quats hemisphere-consistent
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    oi = np.stack([np.interp(t_new, t_src, c) for c in np.concatenate([o[:, :3], quat, o[:, 7:]], 1).T], 1)
    oi[:, 3:7] /= np.linalg.norm(oi[:, 3:7], axis=1, keepdims=True)
    return qi, oi


def set_frame(env, q, o):
    robot, obj = env.robot, env.object
    dev = env.device
    env_idx = torch.tensor([0], device=dev)
    robot.entity.set_dofs_position(
        position=torch.as_tensor(q, device=dev, dtype=torch.float32)[None],
        dofs_idx_local=robot.actuated_dof_idxs, zero_velocity=True, envs_idx=env_idx,
    )
    ot = torch.as_tensor(o, device=dev, dtype=torch.float32)
    obj.set_object_state(ot[:3][None], ot[3:7][None], ot[7:][None] if obj.num_joints > 0 else None, env_idxs=env_idx)
    env.scene.visualizer.update(force=True)


def open_writer(path, fps, crf):
    return imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=None,
        ffmpeg_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sid", required=True, help="e.g. gregen_00025 (reads retarget_logs/<sid>.out)")
    ap.add_argument("--view", default="both", choices=["wide", "square", "both", "probe"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slowmo", type=int, default=1, help="interpolation factor (2 = half speed at the same fps)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--frame-dump", action="store_true", help="also save first/mid/last frames as PNG")
    ap.add_argument("--dump-only", action="store_true", help="only save first/mid/last PNGs, no video (camera tuning)")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: render only the first N frames")
    ap.add_argument("--rollout", default=None, help="eval_ep*.npy policy state dump to replay instead of the reference")
    ap.add_argument("--tag", default="", help="extra suffix for the output name")
    ap.add_argument("--override", nargs="*", default=[], help="extra Hydra overrides, e.g. task.obj_cfg.load_kwargs.scale=0.001")
    args = ap.parse_args()

    meta = parse_done_line(args.sid)
    if args.view == "probe":
        views, args.dump_only = PROBE_VIEWS, True
    else:
        views = {k: v for k, v in VIEWS.items() if args.view in (k, "both")}
    args.frame_dump = args.frame_dump or args.dump_only
    print(f"[{args.sid}] obj={meta['obj']} cond={meta['cond']} base={meta['base']} epl={meta['episode_length']} ref={meta['ref']}")

    t0 = time.time()
    env, cams = build_env(meta, views, extra_overrides=args.override)
    env.reset()
    print(f"env built in {time.time() - t0:.1f}s; robot ndof={env.robot.ndof} obj joints={env.object.num_joints}")

    q, o = load_states(env, args.rollout)
    print(f"states: qpos {q.shape} obj {o.shape}  (ref frames -> {len(q) / args.fps:.2f}s at {args.fps} fps)")
    q, o = interpolate(q, o, args.slowmo)
    if args.max_frames > 0:
        q, o = q[: args.max_frames], o[: args.max_frames]
    T = len(q)

    gseg = ground_seg_id(env.scene, env.ground)
    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(args.out, "scratch")
    os.makedirs(scratch, exist_ok=True)

    stem = f"g1_{meta['obj']}_{meta['cond']}"
    if args.rollout:
        stem += "_policy"
    if args.slowmo > 1:
        stem += f"_slowmo{args.slowmo}x"
    if args.tag:
        stem += f"_{args.tag}"
    paths = {name: os.path.join(args.out, f"{stem}_{name}.mp4") for name in views}
    writers = {} if args.dump_only else {name: open_writer(p, args.fps, args.crf) for name, p in paths.items()}
    dump_idx = {0, T // 2, T - 1}
    frame_iter = sorted(dump_idx) if args.dump_only else range(T)

    t0 = time.time()
    for t in frame_iter:
        set_frame(env, q[t], o[t])
        for name, cam in cams.items():
            rgb, _, seg, _ = cam.render(rgb=True, segmentation=True)
            if t == 0:
                print(f"[{name}] seg ids={np.unique(seg).tolist()} ground seg id={gseg} (ground.idx={env.ground.idx})")
                if rgb.max() == 0:
                    sys.exit(f"first frame is all black: EGL context on EGL_DEVICE_ID={os.environ['EGL_DEVICE_ID']} "
                             "is not rendering; try another device id (1/3/6/7 work on this machine)")
            frame = rgb[..., :3].copy()
            frame[(seg == gseg) | (seg == 0)] = 255  # ground plane + background -> pure white (kills the plane's horizon line)
            if name in writers:
                writers[name].append_data(frame)
            if args.frame_dump and t in dump_idx:
                imageio.imwrite(os.path.join(scratch, f"{stem}_{name}_f{t:04d}.png"), frame)
        if t % 30 == 0:
            print(f"  frame {t}/{T}  {time.time() - t0:.1f}s")
    for w in writers.values():
        w.close()
    dt = time.time() - t0
    print(f"rendered {T} frames x {len(views)} views in {dt:.1f}s ({dt / T:.2f}s/frame)")
    for p in (paths.values() if not args.dump_only else []):
        print("wrote", p)


if __name__ == "__main__":
    main()
