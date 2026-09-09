"""
tools/dsec_make_poses.py -- build a camera trajectory (poses.txt, TUM format) for a DSEC
sequence from the released LiDAR bag, WITHOUT ROS:

    velodyne_decoder (bag -> per-scan point clouds)  ->  KISS-ICP (LiDAR odometry)
    ->  T_world_camRect0 via the calibration chain    ->  poses.txt (t[us] x y z qx qy qz qw)

Clock note (verified on zurich_city_04_a): events/disparity use microseconds-since-UTC-
midnight of the recording day; the bag uses UNIX epoch seconds. The script aligns them by
reducing bag stamps modulo 86400 s and checks overlap with the event range, printing both
so a human can sanity-check.

Calibration chain: cam_to_lidar.yaml provides T_lidar_camRect1 (rectified left FRAME cam);
cam_to_cam.yaml provides T_10 and R_rect0/R_rect1, giving
    T_camRect1_camRect0 = R_rect1 * T_10 * R_rect0^{-1}
    T_lidar_camRect0    = T_lidar_camRect1 * T_camRect1_camRect0
    T_world_camRect0    = T_world_lidar * T_lidar_camRect0

Usage:
    python tools/dsec_make_poses.py data/dsec/lidar/data/zurich_city_04 \
        data/dsec/zurich_city_04_a
"""
from __future__ import annotations
import os, sys

import numpy as np
import yaml


def R44(R: np.ndarray) -> np.ndarray:
    M = np.eye(4); M[:3, :3] = R
    return M


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4); q[i] = 1.0
        return q
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([x, y, z, w])


def main(drive_dir: str, seq_dir: str) -> None:
    import h5py
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import velodyne_decoder as vd
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    # ---- event-clock window of the sequence (us since UTC midnight) -------------------
    with h5py.File(os.path.join(seq_dir, "events", "left", "events.h5"), "r") as f:
        t_off = int(f["t_offset"][()])
        ev_lo = (t_off + int(f["events/t"][0])) * 1e-6      # seconds since midnight
        ev_hi = (t_off + int(f["events/t"][-1])) * 1e-6
    print(f"event window (s since UTC midnight): {ev_lo:.1f} .. {ev_hi:.1f}")

    # ---- read + clock-align lidar scans ----------------------------------------------
    # Bag reading uses `rosbags` (pure python, py3.13-safe; the rospypi `rosbag` stack
    # hangs on modern Python). VelodyneScan payloads are parsed manually — each packet is
    # a fixed 8-byte stamp + 1206-byte data block — and decoded via velodyne_decoder's
    # PacketVector API. Only messages inside the sequence window are read (start/stop).
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    bag = os.path.join(drive_dir, "lidar_imu.bag")
    tstore = get_typestore(Stores.ROS1_NOETIC)
    pad = 3.0

    scans, stamps = [], []
    with Reader(bag) as r:
        day_base = (r.start_time * 1e-9 // 86400.0) * 86400.0   # epoch of UTC midnight
        print(f"bag start (s of day): {(r.start_time*1e-9) % 86400:.1f} | "
              f"target window: {ev_lo:.1f}..{ev_hi:.1f}", flush=True)
        start_ns = int((day_base + ev_lo - pad) * 1e9)
        stop_ns = int((day_base + ev_hi + pad) * 1e9)
        # DSEC lidar bags ship /velodyne_points as sensor_msgs/PointCloud2 —
        # point clouds are already decoded, no packet decoding needed
        conns = [c for c in r.connections if c.topic == "/velodyne_points"]
        if not conns:
            raise SystemExit(f"no /velodyne_points topic; topics: "
                             f"{[c.topic for c in r.connections]}")
        for c, t_ns, raw in r.messages(connections=conns, start=start_ns, stop=stop_ns):
            msg = tstore.deserialize_ros1(raw, c.msgtype)
            n = int(msg.width) * int(msg.height)
            step = int(msg.point_step)
            offs = {f.name: int(f.offset) for f in msg.fields}
            buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)
            xyz = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view("<f4")[:, 0]
                            for k in ("x", "y", "z")], axis=1).astype(np.float64)
            stamps.append(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)
            scans.append(xyz)
    if not scans:
        raise SystemExit("no lidar scans overlap the event window — check clock alignment")
    stamps = np.array(stamps)
    day = stamps % 86400.0                                    # UTC seconds-of-day
    idx = np.arange(len(scans))
    print(f"scans overlapping sequence: {len(idx)} | window (s of day): "
          f"{day.min():.1f} .. {day.max():.1f}", flush=True)

    # ---- KISS-ICP odometry -------------------------------------------------------------
    cfg = KISSConfig()
    cfg.data.max_range = 100.0
    cfg.data.deskew = False          # per-point times unused here; scans are short
    cfg.mapping.voxel_size = 1.0     # KISSConfig() leaves this None; 1m ~ max_range/100
    odom = KissICP(cfg)
    poses_lidar = []
    for k, i in enumerate(idx):
        pts = scans[i]
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64) \
            if pts.dtype.names else pts[:, :3].astype(np.float64)
        m = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.5)
        odom.register_frame(xyz[m], timestamps=np.zeros(int(m.sum())))
        poses_lidar.append(odom.last_pose if hasattr(odom, "last_pose")
                           else odom.poses[-1])
        if (k + 1) % 50 == 0:
            print(f"  odometry {k+1}/{len(idx)}")
    poses_lidar = np.stack(poses_lidar)                      # T_world_lidar (4,4) each

    # ---- calibration chain to camRect0 -------------------------------------------------
    c2l = yaml.safe_load(open(os.path.join(drive_dir, "cam_to_lidar.yaml")))
    T_lidar_camRect1 = np.array(c2l["T_lidar_camRect1"], dtype=np.float64)
    c2c = yaml.safe_load(open(os.path.join(seq_dir, "calibration", "cam_to_cam.yaml")))
    T_10 = np.array(c2c["extrinsics"]["T_10"], dtype=np.float64)
    R_r0 = np.array(c2c["extrinsics"]["R_rect0"], dtype=np.float64)
    R_r1 = np.array(c2c["extrinsics"]["R_rect1"], dtype=np.float64)
    T_camRect1_camRect0 = R44(R_r1) @ T_10 @ np.linalg.inv(R44(R_r0))
    T_lidar_camRect0 = T_lidar_camRect1 @ T_camRect1_camRect0

    # ---- write poses.txt --------------------------------------------------------------
    # Timestamps in the RAW event clock (us relative to t_offset), i.e. the same axis as
    # events/t that DSECDataset.stream()/_cam_at use for its windows. Cam-to-world.
    out = os.path.join(seq_dir, "poses.txt")
    with open(out, "w") as f:
        for T_wl, s_day in zip(poses_lidar, day[idx]):
            T_wc = T_wl @ T_lidar_camRect0                   # cam-to-world
            q = rot_to_quat_xyzw(T_wc[:3, :3])
            t_us = s_day * 1e6 - t_off                       # raw event clock
            f.write(f"{t_us:.0f} {T_wc[0,3]:.6f} {T_wc[1,3]:.6f} {T_wc[2,3]:.6f} "
                    f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
    print(f"wrote {len(idx)} poses -> {out} (t in us, raw event clock)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
