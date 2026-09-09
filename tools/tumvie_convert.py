"""
tools/tumvie_convert.py -- convert TUM-VIE (Klenk et al., IROS'21) to a DSEC-style bundle
that EvSpike-GS can stream, for the B2/B3 Protocol-A comparison against IncEventGS.

*** STATUS: GROUNDED SCAFFOLD, NOT RUN. ***
The TUM-VIE raw formats below are taken from the dataset page (h5 events; JSON calib;
mocap 6-DoF; grayscale frames) but this box has neither the data nor their toolkit, so the
three parts marked `>>> VERIFY <<<` must be confirmed against their sample loader
(https://tumevent-vi.vision.in.tum.de/access_visualize_h5.py) and the calibration JSON before
a real run. The event->(N,4) conversion and the TUM poses.txt writer are exact.

Why DSEC-style output (not MVSEC): TUM-VIE has **no depth GT** (MVSECDataset needs depth) and is
evaluated in image space (novel-view PSNR), like EventNeRF. It ships an external 6-DoF trajectory
(mocap) + intrinsics, exactly the shape `DSECDataset` already consumes (events + `poses.txt` TUM
rows + intrinsics arg). It therefore needs a THIN `TUMVIEDataset` = DSECDataset minus the
disparity/depth path, plus held-out-frame PSNR eval (see NOTE at the bottom). This converter
produces that loader's inputs.

TUM-VIE raw layout (dataset page):
  <seq>-events_left.h5   : events as tuples (x, y, t_us, polarity); + ms_to_idx (DSEC-style)
                           event cam = Prophesee Gen4, 1280x720, Kannala-Brandt fisheye
  vi_gt_data (.tar.gz)   : stereo grayscale frames @20Hz, 1024x1024, + timestamps
  camera-calibrationA.json : intrinsics of all 4 cams (KB, 8 params) + extrinsics T_imu_camN
  mocap-imu-calibrationA.json : T_imu_marker (mocap<->imu)
  <mocap>.txt            : 6-DoF mocap @120Hz  (>>> VERIFY column layout <<<)

Output bundle (DSEC-style, consumed by `pipeline3d/datasets.py::TUMVIEDataset`):
  <out>/<seq>/
    events.npy         (N,4) float64: x, y, t_seconds, polarity(+1/-1)
    poses.txt          TUM rows "t x y z qx qy qz qw", EVENT-cam-to-world (rectified frame)
    intrinsics.txt     pinhole "fx fy cx cy" AFTER KB->pinhole rectification  (>>> VERIFY <<<)
    -- optional, but REQUIRED for the brightness anchor (Prop. 2) --
    images/frameNNNNNN.png              grayscale FRAME-camera images
    images_ts.txt                       rows "t_seconds frameNNNNNN.png"
    anchor_cam.txt                      "fx fy cx cy H W" of the FRAME camera
    poses_frame.txt                     TUM rows, FRAME-cam-to-world

  The frame camera is NOT the event camera (1024x1024 vs 1280x720), so the anchor is a
  cross-camera term: the field is rendered from the frame camera's own pose+intrinsics and
  compared with its image (no warping -- the radiance field is view-consistent). Without
  these four files TUMVIEDataset disables the anchor and Prop. 2 predicts radiance-scale
  drift (the DSEC negative result, RESULTS B9).

Usage (on the GPU/data box):
    python tools/tumvie_convert.py <tumvie_seq_dir> <seq_name> <out_dir> [--cam left]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np


# ----------------------------------------------------------------------------- events (EXACT)
def load_events_h5(path: str) -> np.ndarray:
    """(x, y, t_us, polarity) -> (N,4) float64 [x, y, t_seconds, +-1]. Timestamps us->s."""
    import h5py
    with h5py.File(path, "r") as f:
        # >>> VERIFY <<< exact dataset keys via access_visualize_h5.py. Two common layouts:
        #   (a) a group 'events' with sub-datasets x/y/t/p, or
        #   (b) separate top-level datasets. Handle (a); adapt if (b).
        g = f["events"] if "events" in f else f
        x = np.asarray(g["x"]).astype(np.float64)
        y = np.asarray(g["y"]).astype(np.float64)
        t_us = np.asarray(g["t"]).astype(np.float64)          # microseconds
        p = np.asarray(g["p"]).astype(np.float64)
    t_s = t_us * 1e-6
    pol = np.where(p > 0, 1.0, -1.0)
    return np.stack([x, y, t_s, pol], axis=1)


# ------------------------------------------------------------------------- calibration (JSON)
def load_calib(json_path: str, cam_key: str):
    """Return (kb_params, T_imu_cam 4x4) for the chosen event camera.
    >>> VERIFY <<< the exact JSON nesting (field names for the KB intrinsics list and the
    T_imu_cam block) against camera-calibrationA.json; the page only guarantees these EXIST."""
    with open(json_path) as f:
        calib = json.load(f)
    # placeholder access paths -- confirm keys against the real file:
    kb = np.asarray(calib[cam_key]["intrinsics"], dtype=np.float64)        # 8 KB params
    T_imu_cam = np.asarray(calib[cam_key]["T_imu_cam"], dtype=np.float64).reshape(4, 4)
    return kb, T_imu_cam


def kb_to_pinhole(kb: np.ndarray, W: int, H: int):
    """KB (fisheye) -> approximate pinhole fx,fy,cx,cy for our PinholeCamera pipeline.
    >>> VERIFY <<< EvSpike-GS renders a pinhole model; TUM-VIE event cam is KB fisheye. Either
    (i) undistort events+frames to a pinhole virtual camera (recommended; cv2.fisheye), then use
    its K, or (ii) crop the low-distortion centre. kb[:2]=fx,fy, kb[2:4]=cx,cy in KB param order
    -- confirm the order in the JSON. Returning the raw first four as a first approximation."""
    fx, fy, cx, cy = float(kb[0]), float(kb[1]), float(kb[2]), float(kb[3])
    return fx, fy, cx, cy


# ------------------------------------------------------------------------------- mocap poses
def load_mocap_c2w(mocap_txt: str, T_imu_marker: np.ndarray, T_imu_cam: np.ndarray):
    """mocap T_world_marker(t) -> EVENT-cam-to-world c2w(t), TUM order out.
    Chain:  T_world_cam = T_world_marker @ inv(T_imu_marker) @ T_imu_cam
    >>> VERIFY <<< (a) mocap file columns (assumed TUM: t x y z qx qy qz qw); (b) that
    T_imu_marker/T_imu_cam are stored as imu<-X (invert accordingly)."""
    from numpy.linalg import inv
    rows = np.loadtxt(mocap_txt)                      # assume (M, 8): t x y z qx qy qz qw
    ts = rows[:, 0]
    out = []
    T_marker_imu = inv(T_imu_marker)
    for r in rows:
        Twm = _tum_to_mat(r[1:])                      # T_world_marker
        Twc = Twm @ T_marker_imu @ T_imu_cam          # T_world_cam (c2w)
        out.append((r[0], _mat_to_tum(Twc)))
    return ts, out


def _tum_to_mat(v: np.ndarray) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = v
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = (x, y, z)
    return M


def _mat_to_tum(M: np.ndarray):
    R = M[:3, :3]; t = M[:3, 3]
    qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = (R[2, 1] - R[1, 2]) / (4*qw + 1e-12)
    qy = (R[0, 2] - R[2, 0]) / (4*qw + 1e-12)
    qz = (R[1, 0] - R[0, 1]) / (4*qw + 1e-12)
    return (t[0], t[1], t[2], qx, qy, qz, qw)


# ------------------------------------------------------------------------------------- driver
def convert(seq_dir: str, seq: str, out_dir: str, cam: str = "left") -> None:
    out = os.path.join(out_dir, seq)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    cam_key = f"cam_event_{cam}"                       # >>> VERIFY <<< key name in the JSON

    ev = load_events_h5(os.path.join(seq_dir, f"{seq}-events_{cam}.h5"))
    np.save(os.path.join(out, "events.npy"), ev)
    print(f"  events: {ev.shape[0]:,} -> events.npy")

    kb, T_imu_cam = load_calib(os.path.join(seq_dir, "camera-calibrationA.json"), cam_key)
    T_imu_marker = np.asarray(
        json.load(open(os.path.join(seq_dir, "mocap-imu-calibrationA.json")))["T_imu_marker"]
    ).reshape(4, 4)                                    # >>> VERIFY <<< key/shape
    fx, fy, cx, cy = kb_to_pinhole(kb, 1280, 720)
    with open(os.path.join(out, "intrinsics.txt"), "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")

    ts, poses = load_mocap_c2w(os.path.join(seq_dir, f"{seq}-mocap.txt"),   # >>> VERIFY name
                               T_imu_marker, T_imu_cam)
    with open(os.path.join(out, "poses.txt"), "w") as f:
        for t, q in poses:
            f.write(f"{t} " + " ".join(f"{v:.9g}" for v in q) + "\n")
    print(f"  poses: {len(poses)} -> poses.txt (TUM, event-cam-to-world)")

    # ---- FRAME camera: anchor files (optional but needed for Prop. 2) --------------------
    # >>> VERIFY <<< unpack vi_gt_data and confirm its internal layout + frame-timestamp file,
    # then emit all four: images/frameNNNNNN.png, images_ts.txt ("t path" per row),
    # anchor_cam.txt ("fx fy cx cy H W" of the FRAME cam, after undistortion) and
    # poses_frame.txt (TUM, FRAME-cam-to-world = same mocap chain with T_imu_cam of the
    # frame camera -- reuse load_calib(cam_key='cam_frame_left') + load_mocap_c2w).
    print("  [TODO] frames/anchor: vi_gt_data -> images/ + images_ts.txt + anchor_cam.txt "
          "+ poses_frame.txt  (without these the anchor is OFF -> Prop.2 scale drift)")
    print(f"  wrote bundle: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir"); ap.add_argument("seq"); ap.add_argument("out_dir")
    ap.add_argument("--cam", default="left", choices=["left", "right"])
    a = ap.parse_args()
    convert(a.seq_dir, a.seq, a.out_dir, a.cam)

# ============================================================================================
# NOTE — the companion loader IS IMPLEMENTED: `pipeline3d/datasets.py::TUMVIEDataset`
#   (events.npy streamed by time windows, poses.txt trajectory, intrinsics.txt fallback,
#   optional cross-camera anchor, `eval_views()` for image-space parity; no depth path since
#   TUM-VIE has no depth GT). Smoke-tested on a synthetic bundle; the 23 unit tests still pass.
#   Run:
#     python pipeline3d/train_real.py --dataset tumvie --root <out_dir> --scene <seq> \
#         --theta 0.02 --budget 20000 --device cuda        # --intrinsics optional
#   then compare head-to-head with IncEventGS on the SAME sequence and the SAME held-out
#   frames (RUNBOOK_inceventgs_B2.md, Protocol A). Closes B2 (baseline) + B3 (dataset) together.
#   REMAINING BLOCKER is this file's `>>> VERIFY <<<` items (real TUM-VIE data required).
# ============================================================================================
