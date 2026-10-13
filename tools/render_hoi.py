#!/usr/bin/env python
"""
Headless renderer for EmbodyHOI website clips: MANO hands + articulated ARCTIC
object (+ optional ghosted Unitree G1) from sample .npz dumps, comparing
IK-guided vs unguided generations.

Usage:
  CUDA_VISIBLE_DEVICES=0 python render_hoi.py --robot g1 --idx 25 --idx 10 \
      --mode overlay --with-robot --res 1080 --fps 10 --out media/hoi

Modes:
  overlay     object + guided hands (solid) + unguided hands (translucent)
              [+ robot ghost of the guided IK solution with --with-robot]
  sidebyside  [Unguided | IK-guided] side by side, labels burned in
  single      one npz (guided unless --unguided)
"""
import numpy as np

# NumPy 2.x compatibility for legacy chumpy/smplx stack (must precede MANO imports)
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "unicode"):
    np.unicode = str
if not hasattr(np, "str"):
    np.str = str

import argparse
import os
import subprocess
import sys

R = "/edrive1/kartiksh/MAD-HOI_IK"
HOI_MARDM = "/scr/kartiksh/hoi_mardm"
sys.path.insert(0, R)
sys.path.insert(0, HOI_MARDM)
_CWD = os.getcwd()
os.chdir(HOI_MARDM)  # lib.models.mano loads configs/mano.yaml relative to cwd

import imageio.v3 as iio
import torch
import yourdfpy
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings,
                                look_at_view_transform)
from pytorch3d.structures import Meshes

from lib.utils.proc_output import get_hand_verts  # noqa: E402
from lib.models.mano import build_mano_aa  # noqa: E402
from helper import load_obj_from_name  # noqa: E402

OBJ_ROOTS = {"arctic": "/scr/kartiksh/hoi_mardm/data/arctic/object_vtemplates"}
URDF_G1 = (f"{R}/dexmachinahumanoid/dexmachina/assets/g1/"
           "g1_body29_frozen_hand_mod_waist_new_arms_waist_yaw_lower_frozen.urdf")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SAMPLE_DIRS = {
    "g1": {
        "guided": [f"{R}/ablation_runs/g1/ik_rest_w_10_ik_inner_manip_w_0p03",
                   f"{R}/outputs/icra/g1_guided_extra"],
        "unguided": [f"{R}/ablation_runs/g1/unguided"],
        "joints_guided": f"{R}/outputs/icra/limits/joints_guided58",
        "joints_unguided": f"{R}/outputs/icra/scores58/joints_unguided_floatbase",
    },
    "h12": {
        "guided": [f"{R}/outputs/icra/h12full_guided"],
        "unguided": [f"{R}/outputs/icra/h12full_unguided"],
    },
}

COL_OBJ = (0.72, 0.78, 0.88)
COL_OBJ_TOP = (0.58, 0.65, 0.78)
COL_UNG = {"l": (0.35, 0.80, 0.85), "r": (0.35, 0.75, 0.40)}
COL_GUI = {"l": (0.95, 0.80, 0.25), "r": (0.85, 0.35, 0.70)}
COL_ROBOT = (0.60, 0.60, 0.65)
ALPHA_UNG = 0.60
ALPHA_ROBOT = 0.45

# Robot links not drawn in the ghost (legs are out of frame / distracting)
ROBOT_SKIP = ("hip", "knee", "ankle", "pelvis_contour")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def find_sample(robot, kind, idx):
    for d in SAMPLE_DIRS[robot][kind]:
        p = os.path.join(d, f"sample_{idx:05d}_r00.npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{robot}/{kind} sample {idx}")


class Loader:
    def __init__(self, device):
        self.device = device
        self.layers = {"l": build_mano_aa(is_rhand=False, flat_hand=False).to(device),
                       "r": build_mano_aa(is_rhand=True, flat_hand=False).to(device)}
        self.hand_faces = {k: np.asarray(v.faces, dtype=np.int64) for k, v in self.layers.items()}
        self.obj_cache = {}

    def obj_faces(self, name):
        if name not in self.obj_cache:
            _, faces, top_idx = load_obj_from_name(name, "arctic", self.device, None, OBJ_ROOTS, None)
            self.obj_cache[name] = (np.asarray(faces, dtype=np.int64),
                                    top_idx[0].cpu().numpy().astype(bool))
        return self.obj_cache[name]

    def load(self, path):
        d = np.load(path, allow_pickle=True)
        T = int(d["x_obj"].shape[1])
        out = {"text": str(d["text"]), "object_name": str(d["object_name"]), "T": T,
               "obj_verts": d["obj_vert_traj"][0, :T].astype(np.float32), "hands": {}, "trans": {}}
        out["obj_faces"], out["obj_top"] = self.obj_faces(out["object_name"])
        for side, key, flag in (("l", "x_lhand", "is_lhand"), ("r", "x_rhand", "is_rhand")):
            if not bool(d[flag]):
                continue
            params = torch.as_tensor(d[key][:, :T], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v = get_hand_verts(params, self.layers[side])[0]
            out["hands"][side] = v.cpu().numpy()
            out["trans"][side] = d[key][0, :T, :3]
        return out


# ----------------------------------------------------------------------------
# Robot forward kinematics (yourdfpy)
# ----------------------------------------------------------------------------
def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cluster_decimate(v, f, cell):
    """Vertex-clustering decimation: merge vertices within `cell`-sized voxels."""
    q = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    nv = np.zeros((inv.max() + 1, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv)[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    return nv, nf[keep]


class RobotFK:
    def __init__(self, urdf_path, base_quat_wxyz=(-0.707, 0, 0, 0.707), cell=0.004):
        self.urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=True, load_meshes=True)
        self.base_R = quat_wxyz_to_mat(np.asarray(base_quat_wxyz, dtype=np.float64))
        scene = self.urdf.scene
        self.geoms = []  # (node, vertices (V,3), faces (F,3))
        for node in scene.graph.nodes_geometry:
            _, gname = scene.graph.get(node)
            if any(s in node for s in ROBOT_SKIP) or any(s in gname for s in ROBOT_SKIP):
                continue
            g = scene.geometry[gname]
            v, f = cluster_decimate(np.asarray(g.vertices, dtype=np.float64),
                                    np.asarray(g.faces, dtype=np.int64), cell)
            self.geoms.append((node, v, f))
        # Concatenated face array (constant across frames)
        faces, off = [], 0
        for _, v, f in self.geoms:
            faces.append(f + off)
            off += len(v)
        self.faces = np.concatenate(faces)

    def mesh_verts(self, cfg, root_t):
        self.urdf.update_cfg(np.asarray(cfg, dtype=np.float64))
        out = []
        for node, v, _ in self.geoms:
            T, _ = self.urdf.scene.graph.get(node)  # geometry -> base_link (pelvis)
            vw = v @ T[:3, :3].T + T[:3, 3]
            out.append(vw @ self.base_R.T + root_t)
        return np.concatenate(out).astype(np.float32)

    def ee_positions(self, cfg, root_t):
        """World position of the IK target links (wrist_ee_link), keyed 'l'/'r'."""
        self.urdf.update_cfg(np.asarray(cfg, dtype=np.float64))
        res = {}
        for side in ("left", "right"):
            T = self.urdf.get_transform(f"{side}_wrist_ee_link", "pelvis")
            res[side[0]] = T[:3, 3] @ self.base_R.T + root_t
        return res


def load_joints(robot, kind, idx):
    p = os.path.join(SAMPLE_DIRS[robot][f"joints_{kind}"], f"sample_{idx:05d}_r00_joints.npz")
    j = np.load(p, allow_pickle=True)
    return j["joint_cfgs"].astype(np.float64), j["root_translations"].astype(np.float64)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
class Renderer:
    """Single-pass layered renderer: rasterize with K faces per pixel, Phong-shade every
    layer, then alpha-composite front to back (translucent parts need no extra passes)."""

    def __init__(self, device, res, at, eye, fov=30.0, ss=2, K=1):
        self.device, self.res, self.ss, self.K = device, res, ss, K
        at, eye = np.asarray(at, np.float64), np.asarray(eye, np.float64)
        Rm, Tm = look_at_view_transform(eye=[tuple(eye)], at=[tuple(at)], up=((0, 0, 1),))
        self.cameras = FoVPerspectiveCameras(device=device, R=Rm, T=Tm, fov=fov, znear=0.05, zfar=10.0)
        raster = RasterizationSettings(image_size=res * ss, blur_radius=0.0, faces_per_pixel=K,
                                       bin_size=128, max_faces_per_bin=30000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster)
        self.eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
        # Soft studio lights defined in the camera frame: key above-left, fill right, rim behind.
        fwd = (at - eye) / np.linalg.norm(at - eye)
        right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.ambient = 0.42
        self.lights = []  # (position, diffuse, specular)
        for off, dif, spc in (((-1.2, 1.6, -1.6), 0.55, 0.15), ((1.8, 0.4, -1.2), 0.28, 0.05),
                              ((0.4, 1.4, 1.8), 0.22, 0.05)):
            pos = at + off[0] * right + off[1] * up + off[2] * fwd
            self.lights.append((torch.as_tensor(pos, dtype=torch.float32, device=device), dif, spc))

    def render(self, parts):
        """parts: list of (verts (V,3), faces (F,3), rgb, alpha). Returns rgb (H,W,3) float."""
        dev = self.device
        verts, faces, cols, alphas, groups, off = [], [], [], [], [], 0
        for gi, (v, f, c, a) in enumerate(parts):
            verts.append(v); faces.append(f + off); off += len(v)
            cols.append(np.broadcast_to(np.asarray(c, np.float32), (len(v), 3)))
            alphas.append(np.full(len(f), a, np.float32)); groups.append(np.full(len(f), gi, np.int64))
        V = torch.as_tensor(np.concatenate(verts), dtype=torch.float32, device=dev)
        F = torch.as_tensor(np.concatenate(faces), dtype=torch.int64, device=dev)
        C = torch.as_tensor(np.concatenate(cols), dtype=torch.float32, device=dev)
        A = torch.as_tensor(np.concatenate(alphas), device=dev)
        G = torch.as_tensor(np.concatenate(groups), device=dev)
        with torch.no_grad():
            mesh = Meshes(verts=[V], faces=[F])
            frags = self.rasterizer(mesh)
            p2f = frags.pix_to_face[0]                                   # (H,W,K)
            valid = p2f >= 0
            fidx = p2f.clamp(min=0)
            attr = torch.cat([V[F], mesh.verts_normals_packed()[F], C[F]], -1)  # (F,3,9)
            pix = interpolate_face_attributes(frags.pix_to_face, frags.bary_coords, attr)[0]  # (H,W,K,9)
            pos, nrm, col = pix[..., :3], pix[..., 3:6], pix[..., 6:9]
            view = torch.nn.functional.normalize(self.eye - pos, dim=-1)
            nrm = torch.nn.functional.normalize(nrm, dim=-1)
            nrm = torch.where((nrm * view).sum(-1, keepdim=True) < 0, -nrm, nrm)  # two-sided
            shade = col * self.ambient
            for lpos, dif, spc in self.lights:
                ldir = torch.nn.functional.normalize(lpos - pos, dim=-1)
                ndl = (nrm * ldir).sum(-1, keepdim=True).clamp(min=0)
                refl = 2 * ndl * nrm - ldir
                rdv = (refl * view).sum(-1, keepdim=True).clamp(min=0)
                shade = shade + col * dif * ndl + spc * rdv.pow(24.0)
            alpha = A[fidx] * valid
            grp = G[fidx]
            # A translucent object should not darken itself: ignore a layer whose group
            # equals the previous (nearer) layer's group.
            if self.K > 1:
                same = (grp[..., 1:] == grp[..., :-1]) & (alpha[..., :-1] < 1) & valid[..., 1:]
                alpha = torch.cat([alpha[..., :1], alpha[..., 1:] * (~same)], -1)
            trans = torch.ones_like(alpha[..., 0])
            out = torch.zeros_like(shade[:, :, 0, :])
            for k in range(self.K):
                out = out + (trans * alpha[..., k])[..., None] * shade[:, :, k, :]
                trans = trans * (1 - alpha[..., k])
            out = out + trans[..., None]  # white background
            ss = self.ss
            out = out.reshape(self.res, ss, self.res, ss, 3).mean((1, 3)).clamp(0, 1)
        return out.cpu().numpy()


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# Video writing (raw frames -> ffmpeg libx264)
# ----------------------------------------------------------------------------
class VideoWriter:
    def __init__(self, path, w, h, fps, vf=None):
        cmd = ["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_u8):
        self.p.stdin.write(np.ascontiguousarray(frame_u8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def drawtext(text, x=36, y=30, size=44):
    return (f"drawtext=fontfile={FONT}:text='{text}':x={x}:y={y}:fontsize={size}:fontcolor=0x222222:"
            f"box=1:boxcolor=white@0.75:boxborderw=14")


# ----------------------------------------------------------------------------
# Per-clip rendering
# ----------------------------------------------------------------------------
def lerp(a, b, s):
    return a * (1 - s) + b * s


def subframes(T, k):
    """Yield (i0, i1, s) for k-x linear interpolation between native frames."""
    for i in range(T - 1):
        for j in range(k):
            yield i, i + 1, j / k
    yield T - 1, T - 1, 0.0


def hand_parts(sample, cols, i0, i1, s, faces, alpha=1.0):
    parts = []
    for side, v in sample["hands"].items():
        parts.append((lerp(v[i0], v[i1], s), faces[side], cols[side], alpha))
    return parts


def obj_parts(sample, i0, i1, s):
    v = lerp(sample["obj_verts"][i0], sample["obj_verts"][i1], s)
    f = sample["obj_faces"]
    top_face = sample["obj_top"][f].sum(1) >= 2
    return [(v, f[~top_face], COL_OBJ, 1.0), (v, f[top_face], COL_OBJ_TOP, 1.0)]


def camera_for(samples, cam_dir, fov, fill=1.0):
    """Frame the union of object + hands over all frames: look at the bbox centre from
    direction `cam_dir`, at a distance such that the bounding sphere fills the view."""
    pts = [smp["obj_verts"][:, ::7].reshape(-1, 3) for smp in samples]
    pts += [v[:, ::5].reshape(-1, 3) for smp in samples for v in smp["hands"].values()]
    pts = np.concatenate(pts)
    at = (pts.min(0) + pts.max(0)) / 2
    radius = np.linalg.norm(pts - at, axis=1).max()
    dist = radius / np.tan(np.radians(fov) / 2) * fill
    d = np.asarray(cam_dir, np.float64); d /= np.linalg.norm(d)
    return at, at + d * dist


def render_clip(args, loader, renderer_cls, device, idx, robot_fk):
    robot = args.robot
    want_gui = not (args.mode == "single" and args.unguided)
    want_ung = not (args.mode == "single" and not args.unguided)
    gui = loader.load(find_sample(robot, "guided", idx)) if want_gui else None
    ung = loader.load(find_sample(robot, "unguided", idx)) if want_ung else None
    ref = gui if gui is not None else ung
    T = ref["T"]
    name = f"hoi_{robot}_{idx:05d}_{ref['object_name']}_{args.mode}"
    if args.with_robot:
        name += "_robot"
    print(f"[{name}] T={T} '{ref['text']}'", flush=True)

    if gui is not None and ung is not None:
        if gui["T"] != ung["T"]:
            print(f"  [SKIP] guided/unguided length mismatch ({gui['T']} vs {ung['T']})", flush=True)
            return None
        obj_diff = np.linalg.norm(gui["obj_verts"] - ung["obj_verts"], axis=-1).max()
        if obj_diff > 0.02:
            msg = f"object trajectories differ by up to {obj_diff:.3f} m"
            if args.mode == "overlay":
                print(f"  [SKIP] {msg}; overlay would be misleading", flush=True)
                return None
            print(f"  [WARN] {msg}", flush=True)
    translucent = args.mode == "overlay" or args.with_robot
    K = 4 if translucent else 1
    if args.mode == "sidebyside":  # each panel framed on its own sample
        rend_u = renderer_cls(device, args.res, *camera_for([ung], args.cam_dir, args.fov, args.fill), fov=args.fov, K=K)
        rend = renderer_cls(device, args.res, *camera_for([gui], args.cam_dir, args.fov, args.fill), fov=args.fov, K=K)
    else:
        at, eye = camera_for([ref], args.cam_dir, args.fov, args.fill)
        rend = renderer_cls(device, args.res, at, eye, fov=args.fov, K=K)
    hf = loader.hand_faces

    joints = None
    if args.with_robot and robot_fk is not None:
        cfgs, roots = load_joints(robot, "guided", idx)
        if len(cfgs) != T:
            print(f"  joints T={len(cfgs)} != {T}; resampling", flush=True)
            ii = np.linspace(0, len(cfgs) - 1, T).round().astype(int)
            cfgs, roots = cfgs[ii], roots[ii]
        joints = (cfgs, roots)
        # Sanity: the IK targets wrist_ee_link at the MANO root translation (x_*hand[..., :3]);
        # report the residual at t0 (expected < 1 cm).
        w = robot_fk.ee_positions(cfgs[0], roots[0])
        for side in gui["hands"]:
            print(f"  t0 {side}: |robot wrist_ee - MANO trans| = "
                  f"{np.linalg.norm(gui['trans'][side][0] - w[side]):.3f} m", flush=True)

    def robot_part(i0, i1, s):
        cfg = lerp(joints[0][i0], joints[0][i1], s)
        rt = lerp(joints[1][i0], joints[1][i1], s)
        return [(robot_fk.mesh_verts(cfg, rt), robot_fk.faces, COL_ROBOT, ALPHA_ROBOT)]

    k = max(1, round(args.out_fps / args.fps))
    hold = int(round(args.out_fps * args.hold))
    frames_total = (T - 1) * k + 1
    W = args.res * (2 if args.mode == "sidebyside" else 1)
    vf = None
    if args.mode == "sidebyside":
        vf = drawtext("Unguided") + "," + drawtext("IK-guided", x=f"{args.res}+36")
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, name + ".mp4")
    vw = VideoWriter(mp4, W, args.res, args.out_fps, vf=vf)
    mid_native = T // 2
    mid_png = None

    for n, (i0, i1, s) in enumerate(subframes(T, k)):
        if args.mode == "sidebyside":
            left = rend_u.render(obj_parts(ung, i0, i1, s) + hand_parts(ung, COL_UNG, i0, i1, s, hf))
            right = rend.render(obj_parts(gui, i0, i1, s) + hand_parts(gui, COL_GUI, i0, i1, s, hf))
            frame = np.concatenate([left, right], axis=1)
        else:
            if args.mode == "overlay":
                parts = (obj_parts(gui, i0, i1, s) + hand_parts(gui, COL_GUI, i0, i1, s, hf)
                         + hand_parts(ung, COL_UNG, i0, i1, s, hf, ALPHA_UNG))
            else:
                smp, cols = (gui, COL_GUI) if gui is not None else (ung, COL_UNG)
                parts = obj_parts(smp, i0, i1, s) + hand_parts(smp, cols, i0, i1, s, hf)
            if joints is not None:
                parts += robot_part(i0, i1, s)
            frame = rend.render(parts)
        u8 = to_u8(frame)
        vw.write(u8)
        if i0 == mid_native and s == 0.0:
            mid_png = u8
        if args.dump_frames and s == 0.0:
            iio.imwrite(os.path.join(args.out, "scratch", f"{name}_f{i0:03d}.png"), u8)
        if n == frames_total - 1:
            for _ in range(hold):
                vw.write(u8)
    vw.close()
    if mid_png is not None:
        iio.imwrite(os.path.join(args.out, name + "_mid.png"), mid_png)
    print(f"  wrote {mp4}", flush=True)
    return mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["g1", "h12"], required=True)
    ap.add_argument("--idx", type=int, action="append", required=True)
    ap.add_argument("--mode", choices=["overlay", "sidebyside", "single"], default="overlay")
    ap.add_argument("--unguided", action="store_true", help="single mode: render the unguided npz")
    ap.add_argument("--with-robot", action="store_true")
    ap.add_argument("--res", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=10.0, help="native sample rate")
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold last frame")
    ap.add_argument("--fov", type=float, default=30.0)
    ap.add_argument("--cam-dir", type=float, nargs=3, default=(1.0, 0.45, 0.55),
                    help="direction from scene centre to camera (z-up world)")
    ap.add_argument("--fill", type=float, default=0.95, help="<1 zooms in, >1 zooms out")
    ap.add_argument("--base-quat", type=float, nargs=4, default=(-0.707, 0, 0, 0.707),
                    help="pelvis rotation quaternion wxyz")
    ap.add_argument("--dump-frames", action="store_true", help="save native frames to <out>/scratch")
    ap.add_argument("--out", default="/scr/kartiksh/embodyhoi-website/media/hoi")
    args = ap.parse_args()
    args.out = os.path.join(_CWD, args.out)

    device = torch.device("cuda")
    loader = Loader(device)
    robot_fk = None
    if args.with_robot:
        if args.robot != "g1":
            raise SystemExit("--with-robot is only supported for g1")
        robot_fk = RobotFK(URDF_G1, args.base_quat)
    if args.dump_frames:
        os.makedirs(os.path.join(args.out, "scratch"), exist_ok=True)
    for idx in args.idx:
        try:
            render_clip(args, loader, Renderer, device, idx, robot_fk)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)


if __name__ == "__main__":
    main()
