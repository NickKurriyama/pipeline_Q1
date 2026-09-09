"""
pipeline3d/datasets.py  --  real dataset loaders (EventNeRF, DSEC).

Two-tier dataset strategy (see the paper's experimental protocol):
  - EventNeRF-synthetic (object-centric): head-to-head comparison vs event-3DGS baselines,
    valid Chamfer/geometry because clean GT exists.
  - DSEC (driving, streaming): demonstrate ONLINE streaming reconstruction + depth-vs-LiDAR.

Both yield a common interface consumed by trainer3d.EvSpikeGSTrainer3D:

    for packet in loader.stream():
        packet.dE          # (H*W,) accumulated event map  sum(p*C)  for the window
        packet.cam_prev    # PinholeCamera at window start
        packet.cam_curr    # PinholeCamera at window end
        packet.keyframe    # optional {"cam": ..., "image": ...} anchor, or None

    trainer.streaming_step_events(packet.cam_prev, packet.cam_curr, packet.dE,
                                  anchor=packet.keyframe)

NOTE: written against the published formats of r00tman/EventNeRF and DSEC; verify
field names against the actual downloaded data before a full run (formats have minor
per-release variations). h5py is required only for DSEC (`pip install h5py`).
"""
from __future__ import annotations
import os, glob
from dataclasses import dataclass
from typing import Optional, Iterator, Any

import numpy as np
import torch

from render3d import PinholeCamera


@dataclass
class EventPacket:
    dE: Any                       # (H*W,) torch tensor, accumulated p*C per pixel
    cam_prev: Any
    cam_curr: Any
    keyframe: Optional[Any] = None


# ------------------------------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------------------------------
def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Spherical interpolation between two wxyz quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:  # rare; fall back to largest diagonal branch
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[i + 1] = 1.0
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


class PoseTrack:
    """Camera-to-world trajectory keyed by normalized time in [0,1]; slerp/lerp lookup."""

    def __init__(self, times: np.ndarray, Rs_c2w: np.ndarray, ts_c2w: np.ndarray):
        self.times = times
        self.quats = np.stack([_R_to_quat(R) for R in Rs_c2w])
        self.trans = ts_c2w

    def world_to_cam(self, t: float):
        """Interpolated pose at normalized time t; returns (R, t) world->camera."""
        i = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2))
        t0, t1 = self.times[i], self.times[i + 1]
        a = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
        q = _slerp(self.quats[i], self.quats[i + 1], np.clip(a, 0, 1))
        R_c2w = _quat_to_R(q)
        t_c2w = (1 - a) * self.trans[i] + a * self.trans[i + 1]
        R_w2c = R_c2w.T
        return R_w2c, -R_w2c @ t_c2w


def _accumulate(xs, ys, ps, H, W, C) -> torch.Tensor:
    """Accumulate events into a per-pixel map sum(p*C), flattened to (H*W,)."""
    dE = np.zeros(H * W, dtype=np.float32)
    np.add.at(dE, ys.astype(np.int64) * W + xs.astype(np.int64),
              ps.astype(np.float32) * C)
    return torch.from_numpy(dE)


def _load_tum_track(path: str):
    """Read a TUM trajectory file ("t x y z qx qy qz qw", camera-to-world) into a
    PoseTrack keyed by normalized time. Returns (track, t_first, t_last) in the file's
    own clock, so callers can map absolute timestamps onto [0,1]."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    times, trans, qxyzw = rows[:, 0], rows[:, 1:4], rows[:, 4:8]
    quats_wxyz = np.concatenate([qxyzw[:, 3:4], qxyzw[:, :3]], axis=1)
    Rs = np.stack([_quat_to_R(q / np.linalg.norm(q)) for q in quats_wxyz])
    t0, t1 = float(times.min()), float(times.max())
    return PoseTrack((times - t0) / max(t1 - t0, 1e-9), Rs, trans), t0, t1


# ------------------------------------------------------------------------------------------
# EventNeRF (r00tman/EventNeRF): object-centric 360°, synthetic (ESIM) + real
# ------------------------------------------------------------------------------------------
class EventNeRFDataset:
    """
    Loads the REAL EventNeRF release format (NeRF++-style, verified against
    r00tman/EventNeRF data_loader_split.py and the downloaded data.tar.gz):

        <root>/<scene>/<split>/
            events/*.npz        one archive, fields x, y, t, p
            pose/r_XXXXX.txt    4x4 camera-to-world (OpenCV convention, +z forward)
            intrinsics/r_XXXXX.txt   4x4 or 5x4 K matrix
            rgb/*.png           sparse RGB views (used only as the brightness anchor)

    The camera rotates at constant rate, so pose index i maps linearly to normalized
    time i / max_index; poses between indices are slerp-interpolated. The paper's
    contrast threshold for the synthetic scenes is C = 0.25 (configs/nerf/*.txt).

    Note: the synthetic events are simulated through a Bayer color filter; we supervise
    a grayscale brightness field, which adds per-pixel chroma noise the Huber loss must
    absorb — stated honestly rather than hidden.
    """

    def __init__(self, root: str, scene: str, split: str = "train",
                 n_windows: int = 200, C: float = 0.25,
                 anchor_every: int = 25):
        self.dir = os.path.join(root, scene, split)
        self.n_windows, self.C, self.anchor_every = n_windows, C, anchor_every
        self.x, self.y, self.t, self.p = self._load_events()
        self.track = self._load_poses()
        self.fx, self.fy, self.cx, self.cy = self._load_intrinsics()
        self.W = int(round(2 * self.cx))
        self.H = int(round(2 * self.cy))
        # guard against events outside the inferred sensor plane
        self.W = max(self.W, int(self.x.max()) + 1)
        self.H = max(self.H, int(self.y.max()) + 1)
        self.anchors = self._load_anchors()
        # giu ds.anchor de tuong thich nguoc (vd render_images.py dung ds.anchor["image"]
        # de estimate_background) -- nhung streaming se dung TOAN BO self.anchors ben duoi
        self.anchor = self.anchors[0] if self.anchors else None
        # green Bayer sites (RGGB/BGGR diagonal): supervising a GRAYSCALE field on
        # color-filtered events is only consistent on one channel — green is the
        # luminance-closest and has 2x the sites of R/B
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        self.green_mask = torch.from_numpy(((xs + ys) % 2 == 1).reshape(-1))

    # ---- pieces ------------------------------------------------------------------
    def _load_events(self):
        files = sorted(glob.glob(os.path.join(self.dir, "events", "*.npz")))
        if not files:
            raise FileNotFoundError(f"no events/*.npz under {self.dir}")
        d = np.load(files[0])
        x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        p = np.where(p > 0, 1, -1).astype(np.int8)
        order = np.argsort(t)
        return x[order], y[order], t[order].astype(np.float64), p[order]

    def _pose_files(self):
        files = sorted(glob.glob(os.path.join(self.dir, "pose", "r_*.txt")))
        if not files:
            raise FileNotFoundError(f"no pose/r_*.txt under {self.dir}")
        idx = [int(os.path.basename(f)[2:-4]) for f in files]
        return [f for _, f in sorted(zip(idx, files))], sorted(idx)

    def _load_poses(self) -> PoseTrack:
        files, idx = self._pose_files()
        mats = np.stack([np.loadtxt(f).reshape(4, 4) for f in files])
        times = np.array(idx, dtype=np.float64) / max(idx[-1], 1)
        return PoseTrack(times, mats[:, :3, :3], mats[:, :3, 3])

    def _load_intrinsics(self):
        files = sorted(glob.glob(os.path.join(self.dir, "intrinsics", "r_*.txt")))
        if not files:
            raise FileNotFoundError(f"no intrinsics/r_*.txt under {self.dir}")
        vals = np.loadtxt(files[0]).reshape(-1)
        K = vals[:16].reshape(4, 4)
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    def _load_anchors(self):
        """TOAN BO cac RGB view co san (grayscale) + camera tuong ung. Voi dataset co
        RGB DAY DAC (gan nhu moi pose deu co anh, nhu EventNeRF ban tai ve thuc te --
        ~1000 anh chu khong "sparse" nhu docstring cu ghi), ta luu ca thoi gian chuan
        hoa (tn) cua tung anchor de tra cuu anchor GAN NHAT VOI GOC HIEN TAI trong
        stream(), thay vi chi lay dung 1 anh dau tien (bug cu) hay xoay vong tuan tu
        (bug cua patch truoc: chi bao giv dung vai anchor dau danh sach, vi
        k // anchor_every << n_anchors nen phep % khong bao gio wrap)."""
        rgbs = sorted(glob.glob(os.path.join(self.dir, "rgb", "*.png")))
        if not rgbs:
            self._anchor_times = np.zeros(0)
            return []
        import matplotlib.image as mpimg
        _, all_idx = self._pose_files()
        anchors, times = [], []
        for rf in rgbs:
            img = mpimg.imread(rf)
            if img.ndim == 3:
                img = img[..., :3].mean(-1)
            idx = int(os.path.basename(rf)[2:-4])
            tn = idx / max(all_idx[-1], 1)
            anchors.append({"cam": self._cam_at(tn),
                            "image": torch.from_numpy(np.ascontiguousarray(img)).float()})
            times.append(tn)
        self._anchor_times = np.array(times)
        return anchors

    def anchor_at(self, tn: float):
        """Anchor (RGB view + camera) co goc/thoi gian GAN NHAT voi tn (trong [0,1]).
        Day la ham quan trong nhat cua patch: dam bao anchor neo LUON gan voi goc nhin
        cua cua so hien tai, thay vi mot anchor co dinh hoac xoay vong khong lien quan
        goc nhin."""
        if not self.anchors:
            return None
        i = int(np.argmin(np.abs(self._anchor_times - tn)))
        return self.anchors[i]

    def _cam_at(self, tn: float) -> PinholeCamera:
        R, t = self.track.world_to_cam(float(np.clip(tn, 0.0, 1.0)))
        return PinholeCamera(self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                             torch.tensor(R, dtype=torch.float32),
                             torch.tensor(t, dtype=torch.float32))

    def stream(self) -> Iterator[EventPacket]:
        t0, t1 = float(self.t.min()), float(self.t.max())
        span = max(t1 - t0, 1e-9)
        edges = np.linspace(t0, t1, self.n_windows + 1)
        starts = np.searchsorted(self.t, edges[:-1], side="left")
        stops = np.searchsorted(self.t, edges[1:], side="right")
        for k, (lo, hi) in enumerate(zip(starts, stops)):
            if hi <= lo:
                continue
            sl = slice(lo, hi)
            dE = _accumulate(self.x[sl], self.y[sl], self.p[sl], self.H, self.W, self.C)
            keyframe = None
            if k % self.anchor_every == 0:
                tn_k = (edges[k] - t0) / span
                keyframe = self.anchor_at(tn_k)
            yield EventPacket(dE=dE,
                              cam_prev=self._cam_at((edges[k] - t0) / span),
                              cam_curr=self._cam_at((edges[k + 1] - t0) / span),
                              keyframe=keyframe)


# ------------------------------------------------------------------------------------------
# MVSEC (indoor_flying / outdoor_day): DAVIS 346x260, GT 6-DoF pose + GT depth in gt.hdf5
# ------------------------------------------------------------------------------------------
class MVSECDataset:
    """
    Loads the MVSEC HDF5 release (daniilidis-group/mvsec):

        <root>/<seq>_data.hdf5 : davis/left/{events (N,4: x,y,t[s],p), image_raw,
                                 image_raw_ts}  (frames used as the brightness anchor)
        <root>/<seq>_gt.hdf5   : davis/left/{pose (M,4,4), pose_ts,
                                 depth_image_raw, depth_image_raw_ts}

    This is the tier with REAL sensor data + GT pose + GT depth, so the full streaming
    pipeline (reconstruction + depth-vs-GT evaluation) runs end to end. Intrinsics are
    passed in (from the calib zip); lens distortion is ignored (stated approximation).
    `pose` is auto-probed for world->cam vs cam->world using the depth maps.
    """

    def __init__(self, root: str, seq: str = "indoor_flying1",
                 intrinsics=(226.38, 226.15, 173.65, 133.74), H=260, W=346,
                 n_windows: int = 400, C: float = 0.25, anchor_every: int = 25,
                 t_start_s: Optional[float] = None, t_stop_s: Optional[float] = None,
                 pose_is_cam_to_world: bool = True):
        import h5py
        self.H, self.W, self.C = H, W, C
        self.n_windows, self.anchor_every = n_windows, anchor_every
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.c2w = pose_is_cam_to_world

        d = h5py.File(os.path.join(root, f"{seq}_data.hdf5"), "r")
        g = h5py.File(os.path.join(root, f"{seq}_gt.hdf5"), "r")
        ev = d["davis"]["left"]["events"][()]              # (N,4) x,y,t,p
        self.pose = g["davis"]["left"]["pose"][()]         # (M,4,4)
        self.pose_ts = g["davis"]["left"]["pose_ts"][()]
        self.depth = g["davis"]["left"]["depth_image_raw"]
        self.depth_ts = g["davis"]["left"]["depth_image_raw_ts"][()]
        self.frames = d["davis"]["left"]["image_raw"]
        self.frames_ts = d["davis"]["left"]["image_raw_ts"][()]

        # clip to the pose-covered interval (and optional user window);
        # t_start/t_stop below 1e6 are treated as OFFSETS from the stream start
        base = max(float(ev[:, 2].min()), float(self.pose_ts.min()))
        def _abs(t):
            return None if t is None else (base + t if t < 1e6 else t)
        t_start_s, t_stop_s = _abs(t_start_s), _abs(t_stop_s)
        lo = max(ev[:, 2].min(), self.pose_ts.min(),
                 t_start_s if t_start_s is not None else -np.inf)
        hi = min(ev[:, 2].max(), self.pose_ts.max(),
                 t_stop_s if t_stop_s is not None else np.inf)
        m = (ev[:, 2] >= lo) & (ev[:, 2] <= hi)
        ev = ev[m]
        order = np.argsort(ev[:, 2])
        self.x = ev[order, 0].astype(np.int32)
        self.y = ev[order, 1].astype(np.int32)
        self.t = ev[order, 2]
        self.p = np.where(ev[order, 3] > 0, 1, -1).astype(np.int8)
        self.t0, self.t1 = float(lo), float(hi)

        qs = np.stack([_R_to_quat(P[:3, :3]) for P in self.pose])
        tn = (self.pose_ts - self.pose_ts.min()) / max(
            self.pose_ts.max() - self.pose_ts.min(), 1e-12)
        self.track = PoseTrack(tn, np.stack([_quat_to_R(q) for q in qs]),
                               self.pose[:, :3, 3])
        self._pose_span = (self.pose_ts.min(), self.pose_ts.max())

    def _cam_at(self, t_s: float) -> PinholeCamera:
        a, b = self._pose_span
        tn = float(np.clip((t_s - a) / max(b - a, 1e-12), 0, 1))
        if self.c2w:
            R, t = self.track.world_to_cam(tn)
        else:   # stored pose already world->cam
            i = int(np.clip(np.searchsorted(self.track.times, tn) - 1,
                            0, len(self.track.times) - 1))
            P = self.pose[i]
            R, t = P[:3, :3], P[:3, 3]
        return PinholeCamera(self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                             torch.tensor(R, dtype=torch.float32),
                             torch.tensor(t, dtype=torch.float32))

    def anchor_at(self, t_s: float):
        """Nearest grayscale DAVIS frame + its camera (brightness anchor)."""
        i = int(np.argmin(np.abs(self.frames_ts - t_s)))
        img = self.frames[i].astype(np.float32) / 255.0
        return {"cam": self._cam_at(float(self.frames_ts[i])),
                "image": torch.from_numpy(img)}

    def depth_at(self, t_s: float):
        """Nearest GT depth map (H,W float, NaN where invalid) + its camera + gap (s)."""
        i = int(np.argmin(np.abs(self.depth_ts - t_s)))
        dep = torch.from_numpy(self.depth[i].astype(np.float32))
        return dep, self._cam_at(float(self.depth_ts[i])), float(
            abs(self.depth_ts[i] - t_s))

    def init_points_from_depth(self, t_s: Optional[float] = None,
                               max_points: int = 20_000) -> torch.Tensor:
        """Back-project the GT depth map nearest to t_s into WORLD points — the coarse
        point-cloud initialisation the protocol prescribes (depth used for init only)."""
        dep, cam, _ = self.depth_at(self.t0 if t_s is None else t_s)
        valid = torch.isfinite(dep) & (dep > 0.1)
        ys, xs = torch.nonzero(valid, as_tuple=True)
        z = dep[valid]
        x_cam = (xs.float() - cam.cx) / cam.fx * z
        y_cam = (ys.float() - cam.cy) / cam.fy * z
        X_cam = torch.stack([x_cam, y_cam, z], dim=1)
        X_world = (X_cam - cam.t[None]) @ cam.R          # R^T applied on the right
        if X_world.shape[0] > max_points:
            idx = torch.randperm(X_world.shape[0])[:max_points]
            X_world = X_world[idx]
        return X_world

    def stream(self) -> Iterator[EventPacket]:
        edges = np.linspace(self.t0, self.t1, self.n_windows + 1)
        starts = np.searchsorted(self.t, edges[:-1], side="left")
        stops = np.searchsorted(self.t, edges[1:], side="right")
        for k, (lo, hi) in enumerate(zip(starts, stops)):
            if hi <= lo:
                continue
            sl = slice(lo, hi)
            dE = _accumulate(self.x[sl], self.y[sl], self.p[sl], self.H, self.W, self.C)
            kf = self.anchor_at(edges[k]) if k % self.anchor_every == 0 else None
            yield EventPacket(dE=dE,
                              cam_prev=self._cam_at(float(edges[k])),
                              cam_curr=self._cam_at(float(edges[k + 1])),
                              keyframe=kf)


# ------------------------------------------------------------------------------------------
# DSEC (driving): left event camera, GT poses supplied as a TUM-style trajectory file
# ------------------------------------------------------------------------------------------
class DSECDataset:
    """
    Expects a sequence directory containing:
      - `events/left/events.h5`      (groups events/{x,y,t,p}; t in microseconds)
      - `events/left/rectify_map.h5` (dataset 'rectify_map', HxWx2)  -- optional
      - a trajectory file `poses.txt` with TUM rows: t x y z qx qy qz qw (camera-to-world),
        e.g. exported from the DSEC ground-truth / your SLAM run.
    Intrinsics (fx fy cx cy) must be passed in (from cam_to_cam.yaml, camRect0).
    """

    def __init__(self, root, sequence, intrinsics, H=480, W=640, window_us=5000, C=0.2,
                 require_poses: bool = True):
        import h5py  # required only for DSEC
        try:
            import hdf5plugin  # noqa: F401  (registers the compression filters DSEC uses)
        except ImportError:
            pass
        self.dir = os.path.join(root, sequence)
        self.H, self.W, self.window_us, self.C = H, W, window_us, C
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.h5 = h5py.File(os.path.join(self.dir, "events", "left", "events.h5"), "r")
        self.ev = self.h5["events"]
        rect = os.path.join(self.dir, "events", "left", "rectify_map.h5")
        self.rectify = None
        if os.path.exists(rect):
            with h5py.File(rect, "r") as f:
                self.rectify = f["rectify_map"][()]          # (H,W,2) float
        # DSEC ships NO ground-truth camera poses (only disparity/LiDAR/IMU/RTK).
        # Reconstruction therefore needs an external trajectory (poses.txt, TUM rows,
        # e.g. from LiDAR-inertial odometry). Without it, require_poses=False enables a
        # THROUGHPUT-ONLY mode: packets stream with an identity camera so the event
        # accumulation/windowing path can be exercised and benchmarked on real data.
        if require_poses or os.path.exists(os.path.join(self.dir, "poses.txt")):
            self.track, self.t0, self.t1 = self._load_poses()
        else:
            self.track = None
            t_all = self.ev["t"]
            self.t0, self.t1 = float(t_all[0]), float(t_all[-1])

    def _load_poses(self):
        rows = np.loadtxt(os.path.join(self.dir, "poses.txt"))
        times = rows[:, 0]
        trans = rows[:, 1:4]
        qxyzw = rows[:, 4:8]
        quats_wxyz = np.concatenate([qxyzw[:, 3:4], qxyzw[:, :3]], axis=1)
        Rs = np.stack([_quat_to_R(q / np.linalg.norm(q)) for q in quats_wxyz])
        t0, t1 = float(times.min()), float(times.max())
        return PoseTrack((times - t0) / max(t1 - t0, 1e-9), Rs, trans), t0, t1

    def _cam_at(self, t_us: float) -> PinholeCamera:
        if self.track is None:      # throughput-only mode (no trajectory available)
            return PinholeCamera(self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                                 torch.eye(3), torch.zeros(3))
        tn = (t_us - self.t0) / max(self.t1 - self.t0, 1e-9)
        R, t = self.track.world_to_cam(float(np.clip(tn, 0, 1)))
        return PinholeCamera(self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                             torch.tensor(R, dtype=torch.float32),
                             torch.tensor(t, dtype=torch.float32))

    # ---- disparity ground truth (train split) ----------------------------------------
    def _load_Q(self):
        """Q matrix for the event-camera pair (disparity_to_depth/cams_03)."""
        import yaml
        cal = yaml.safe_load(open(os.path.join(self.dir, "calibration",
                                               "cam_to_cam.yaml")))
        return np.array(cal["disparity_to_depth"]["cams_03"], dtype=np.float64)

    def disparity_ts_raw(self) -> np.ndarray:
        """Disparity timestamps in the RAW event clock (us, t_offset removed)."""
        ts = np.loadtxt(os.path.join(self.dir, "disparity", "timestamps.txt"),
                        dtype=np.int64)
        t_off = int(self.h5["t_offset"][()])
        return ts - t_off

    def depth_from_disparity(self, k: int):
        """GT depth map (H,W) in meters from the k-th disparity PNG via the Q matrix.
        Zero disparity = invalid -> NaN. Returns (depth, cam at that time, t_raw_us)."""
        import matplotlib.image as mpimg
        path = os.path.join(self.dir, "disparity", f"{2*k:06d}.png")
        disp16 = (mpimg.imread(path) * 65535.0).astype(np.float64) \
            if mpimg.imread(path).dtype != np.uint16 else mpimg.imread(path)
        disp = disp16.astype(np.float64) / 256.0
        Q = self._load_Q()
        # depth Z = (Q[2,3] + Q[2,2]*d)/(Q[3,3] + Q[3,2]*d) per rectified-pair geometry
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = (Q[2, 3] + Q[2, 2] * disp) / (Q[3, 3] + Q[3, 2] * disp)
        Z[disp <= 0] = np.nan
        t_raw = int(self.disparity_ts_raw()[k])
        return torch.from_numpy(np.abs(Z).astype(np.float32)), self._cam_at(t_raw), t_raw

    def init_points_from_disparity(self, k: int = 0,
                                   max_points: int = 50_000) -> torch.Tensor:
        """Back-project the k-th GT disparity map into WORLD points (init only)."""
        dep, cam, _ = self.depth_from_disparity(k)
        valid = torch.isfinite(dep) & (dep > 1.0) & (dep < 80.0)
        ys, xs = torch.nonzero(valid, as_tuple=True)
        z = dep[valid]
        x_cam = (xs.float() - cam.cx) / cam.fx * z
        y_cam = (ys.float() - cam.cy) / cam.fy * z
        X_cam = torch.stack([x_cam, y_cam, z], dim=1)
        X_world = (X_cam - cam.t[None]) @ cam.R
        if X_world.shape[0] > max_points:
            idx = torch.randperm(X_world.shape[0])[:max_points]
            X_world = X_world[idx]
        return X_world

    def stream(self, chunk=2_000_000) -> Iterator[EventPacket]:
        """Iterate the h5 event stream in chunks, cutting fixed-duration windows."""
        n = self.ev["t"].shape[0]
        buf_x = buf_y = buf_t = buf_p = None
        win_lo = None
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            x = self.ev["x"][lo:hi].astype(np.int32)
            y = self.ev["y"][lo:hi].astype(np.int32)
            t = self.ev["t"][lo:hi].astype(np.int64)
            p = self.ev["p"][lo:hi].astype(np.int8)
            p = np.where(p > 0, 1, -1)
            if self.rectify is not None:
                xy = self.rectify[y, x]                       # rectified coords
                x = np.clip(np.round(xy[:, 0]), 0, self.W - 1).astype(np.int32)
                y = np.clip(np.round(xy[:, 1]), 0, self.H - 1).astype(np.int32)
            if buf_t is None:
                buf_x, buf_y, buf_t, buf_p = x, y, t, p
                win_lo = int(t[0])
            else:
                buf_x = np.concatenate([buf_x, x]); buf_y = np.concatenate([buf_y, y])
                buf_t = np.concatenate([buf_t, t]); buf_p = np.concatenate([buf_p, p])
            while buf_t[-1] - win_lo >= self.window_us:
                win_hi = win_lo + self.window_us
                m = buf_t < win_hi
                if m.any():
                    dE = _accumulate(buf_x[m], buf_y[m], buf_p[m], self.H, self.W, self.C)
                    yield EventPacket(dE=dE,
                                      cam_prev=self._cam_at(win_lo),
                                      cam_curr=self._cam_at(win_hi))
                keep = ~m
                buf_x, buf_y, buf_t, buf_p = buf_x[keep], buf_y[keep], buf_t[keep], buf_p[keep]
                win_lo = win_hi
                if buf_t.size == 0:
                    break


# ------------------------------------------------------------------------------------------
# TUM-VIE (Klenk et al., IROS'21): handheld/mocap, event stream + external 6-DoF trajectory
# ------------------------------------------------------------------------------------------
class TUMVIEDataset:
    """
    Consumes the bundle written by `tools/tumvie_convert.py` (raw TUM-VIE h5/JSON/mocap is
    converted once, offline):

        <root>/<seq>/
          events.npy       (N,4) float64: x, y, t_SECONDS, polarity(+-1), time-sorted
          poses.txt        TUM rows "t x y z qx qy qz qw", EVENT-camera-to-world
          intrinsics.txt   "fx fy cx cy"  (pinhole, after the KB->pinhole rectification)
          images/, images_ts.txt, anchor_cam.txt, poses_frame.txt   (OPTIONAL, see below)

    Shape-wise this is the DSEC case (event stream + an external trajectory + intrinsics),
    NOT the MVSEC case: TUM-VIE ships **no depth ground truth**, so evaluation is
    image-space (PSNR/SSIM/LPIPS on held-out frames) as on EventNeRF, never depth-vs-GT.

    ANCHOR (important, Prop. 2). The absolute-brightness anchor needs a real intensity image
    with a known pose. TUM-VIE's grayscale cameras are *not* the event cameras (1024x1024
    frames vs 1280x720 Prophesee), so the anchor is a **cross-camera** term: we render the
    field from the FRAME camera's own pose+intrinsics and compare against its image — no
    warping needed, since the radiance field is view-consistent. That requires the converter
    to also emit `anchor_cam.txt` ("fx fy cx cy H W") and `poses_frame.txt` (frame-cam-to-
    world). If they are absent the anchor is disabled and a warning is printed: without an
    absolute-brightness constraint Prop. 2 predicts the radiance scale drifts — exactly the
    DSEC negative result (RESULTS B9). Do not read a no-anchor run as a method failure.
    """

    def __init__(self, root: str, seq: str, intrinsics=None, H: int = 720, W: int = 1280,
                 n_windows: int = 400, C: float = 0.25, anchor_every: int = 25,
                 t_start_s: Optional[float] = None, t_stop_s: Optional[float] = None):
        self.dir = os.path.join(root, seq)
        self.H, self.W, self.C = H, W, C
        self.n_windows, self.anchor_every = n_windows, anchor_every

        if intrinsics is None:                       # fall back to intrinsics.txt
            f = os.path.join(self.dir, "intrinsics.txt")
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"no intrinsics: pass --intrinsics fx fy cx cy or provide {f}")
            intrinsics = tuple(np.loadtxt(f).ravel()[:4])
        self.fx, self.fy, self.cx, self.cy = (float(v) for v in intrinsics)

        # ---- events: memory-map, bisect the time column, load only the needed slice ----
        ev = np.load(os.path.join(self.dir, "events.npy"), mmap_mode="r")   # (N,4)
        n = ev.shape[0]
        t_first, t_last = float(ev[0, 2]), float(ev[n - 1, 2])
        # half-open [i0, i1); with no clip use the exact array bounds so no event is lost
        i0 = 0 if t_start_s is None else self._bisect_t(ev, max(t_first, float(t_start_s)))
        i1 = n if t_stop_s is None else self._bisect_t(ev, min(t_last, float(t_stop_s)))
        sl = np.asarray(ev[i0:max(i1, i0 + 1)])      # materialise only the clipped window
        self.x = sl[:, 0].astype(np.int32)
        self.y = sl[:, 1].astype(np.int32)
        self.t = sl[:, 2].astype(np.float64)
        self.p = np.where(sl[:, 3] > 0, 1, -1).astype(np.int8)

        # ---- trajectory (event camera) --------------------------------------------------
        self.track, p0, p1 = _load_tum_track(os.path.join(self.dir, "poses.txt"))
        self._pose_span = (p0, p1)
        self.t0 = max(float(self.t[0]), p0)          # stream only where poses exist
        self.t1 = min(float(self.t[-1]), p1)
        if self.t1 <= self.t0:
            raise ValueError(f"events [{self.t[0]:.3f},{self.t[-1]:.3f}] and poses "
                             f"[{p0:.3f},{p1:.3f}] do not overlap — check the clock offset "
                             f"applied in tumvie_convert.py")

        # ---- optional cross-camera anchor (frames from the FRAME camera) -----------------
        self.frames_ts, self.frame_paths, self.anchor_cam_p, self.track_frame = (
            None, None, None, None)
        self._load_frames_if_any()
        self.anchor = self.anchor_at(self.t0) if self.has_anchor else None

    # -------------------------------------------------------------------- helpers
    @staticmethod
    def _bisect_t(ev, target: float) -> int:
        """Binary search on the mmapped time column (O(log N) single-element reads)."""
        lo, hi = 0, ev.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if float(ev[mid, 2]) < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @property
    def has_anchor(self) -> bool:
        return self.frames_ts is not None and self.anchor_cam_p is not None \
            and self.track_frame is not None

    def _load_frames_if_any(self) -> None:
        ts_f = os.path.join(self.dir, "images_ts.txt")
        cam_f = os.path.join(self.dir, "anchor_cam.txt")
        pose_f = os.path.join(self.dir, "poses_frame.txt")
        if not os.path.exists(ts_f):
            print("[TUM-VIE] no images_ts.txt -> NO brightness anchor. Prop. 2 predicts "
                  "radiance-scale drift (cf. DSEC, RESULTS B9); geometry only.")
            return
        rows = [l.split() for l in open(ts_f).read().split("\n") if l.strip()]
        self.frames_ts = np.array([float(r[0]) for r in rows])
        self.frame_paths = [os.path.join(self.dir, "images", r[1]) if len(r) > 1
                            else os.path.join(self.dir, "images", f"frame{i:06d}.png")
                            for i, r in enumerate(rows)]
        if os.path.exists(cam_f) and os.path.exists(pose_f):
            v = np.loadtxt(cam_f).ravel()
            self.anchor_cam_p = (float(v[0]), float(v[1]), float(v[2]), float(v[3]),
                                 int(v[4]), int(v[5]))          # fx fy cx cy H W
            self.track_frame, _, _ = _load_tum_track(pose_f)
        else:
            print(f"[TUM-VIE] frames found but {os.path.basename(cam_f)}/"
                  f"{os.path.basename(pose_f)} missing -> anchor DISABLED (the frame camera "
                  "is not the event camera; its own pose+intrinsics are required).")

    # -------------------------------------------------------------------- cameras
    def _cam_at(self, t_s: float) -> PinholeCamera:
        a, b = self._pose_span
        tn = float(np.clip((t_s - a) / max(b - a, 1e-12), 0, 1))
        R, t = self.track.world_to_cam(tn)
        return PinholeCamera(self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                             torch.tensor(R, dtype=torch.float32),
                             torch.tensor(t, dtype=torch.float32))

    def _frame_cam_at(self, t_s: float) -> PinholeCamera:
        fx, fy, cx, cy, H, W = self.anchor_cam_p
        a, b = self._pose_span
        tn = float(np.clip((t_s - a) / max(b - a, 1e-12), 0, 1))
        R, t = self.track_frame.world_to_cam(tn)
        return PinholeCamera(H, W, fx, fy, cx, cy,
                             torch.tensor(R, dtype=torch.float32),
                             torch.tensor(t, dtype=torch.float32))

    def anchor_at(self, t_s: float):
        """Nearest grayscale FRAME-camera image + that camera (cross-camera anchor)."""
        if not self.has_anchor:
            return None
        i = int(np.argmin(np.abs(self.frames_ts - t_s)))
        import matplotlib.image as mpimg
        img = mpimg.imread(self.frame_paths[i]).astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.max() > 1.5:
            img = img / 255.0
        return {"cam": self._frame_cam_at(float(self.frames_ts[i])),
                "image": torch.from_numpy(img)}

    def eval_views(self, stride: int = 10):
        """Held-out frames for image-space evaluation: [(cam, image), ...].
        Use the SAME held-out list for IncEventGS to keep the comparison honest."""
        out = []
        if not self.has_anchor:
            return out
        for i in range(0, len(self.frames_ts), max(1, stride)):
            a = self.anchor_at(float(self.frames_ts[i]))
            if a is not None:
                out.append((a["cam"], a["image"]))
        return out

    # -------------------------------------------------------------------- streaming
    def stream(self) -> Iterator[EventPacket]:
        edges = np.linspace(self.t0, self.t1, self.n_windows + 1)
        starts = np.searchsorted(self.t, edges[:-1], side="left")
        stops = np.searchsorted(self.t, edges[1:], side="right")
        for k, (lo, hi) in enumerate(zip(starts, stops)):
            if hi <= lo:
                continue
            sl = slice(lo, hi)
            dE = _accumulate(self.x[sl], self.y[sl], self.p[sl], self.H, self.W, self.C)
            kf = self.anchor_at(float(edges[k])) \
                if (self.has_anchor and k % self.anchor_every == 0) else None
            yield EventPacket(dE=dE,
                              cam_prev=self._cam_at(float(edges[k])),
                              cam_curr=self._cam_at(float(edges[k + 1])),
                              keyframe=kf)