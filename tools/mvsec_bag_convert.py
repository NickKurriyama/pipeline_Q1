"""
tools/mvsec_bag_convert.py -- convert MVSEC ROS1 bags to the official HDF5 layout,
WITHOUT a ROS installation (uses the pure-python `rosbags` reader).

The official MVSEC hdf5 release mirrors the bag topics; this produces the same keys the
MVSECDataset loader expects:

    <seq>_data.hdf5 : davis/left/{events (N,4: x,y,t[s],p), image_raw, image_raw_ts}
    <seq>_gt.hdf5   : davis/left/{pose (M,4,4), pose_ts, depth_image_raw,
                                  depth_image_raw_ts}

dvs_msgs/EventArray payloads are parsed directly from the raw ROS1 bytes with numpy
(13 bytes/event: u16 x, u16 y, u32 secs, u32 nsecs, u8 polarity) — orders of magnitude
faster than per-message deserialisation for ~10^7 events.

Usage:
    python tools/mvsec_bag_convert.py <data.bag> <gt.bag> <out_dir> <seq_name>
"""
from __future__ import annotations
import os, sys, struct

import numpy as np
import h5py
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"),
                        ("secs", "<u4"), ("nsecs", "<u4"), ("pol", "u1")])


def parse_event_array(raw: bytes):
    """Manually parse dvs_msgs/EventArray from ROS1 serialized bytes."""
    off = 0
    # std_msgs/Header: uint32 seq, time stamp (2x uint32), string frame_id
    off += 4 + 8
    (slen,) = struct.unpack_from("<I", raw, off); off += 4 + slen
    off += 4 + 4                                   # uint32 height, width
    (n,) = struct.unpack_from("<I", raw, off); off += 4
    ev = np.frombuffer(raw, dtype=EVENT_DTYPE, count=n, offset=off)
    return ev


def convert(data_bag: str, gt_bag: str, out_dir: str, seq: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = get_typestore(Stores.ROS1_NOETIC)

    # ---------------- data bag: events + grayscale frames --------------------------
    ev_chunks, frames, frame_ts = [], [], []
    with Reader(data_bag) as r:
        conns = [c for c in r.connections
                 if c.topic in ("/davis/left/events", "/davis/left/image_raw")]
        for c, t_ns, raw in r.messages(connections=conns):
            if c.topic == "/davis/left/events":
                ev_chunks.append(parse_event_array(raw))
            else:
                msg = ts.deserialize_ros1(raw, c.msgtype)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                frames.append(img.copy())
                frame_ts.append(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)
    ev = np.concatenate(ev_chunks)
    t = ev["secs"].astype(np.float64) + 1e-9 * ev["nsecs"].astype(np.float64)
    events = np.stack([ev["x"].astype(np.float64), ev["y"].astype(np.float64),
                       t, np.where(ev["pol"] > 0, 1.0, -1.0)], axis=1)
    print(f"  events: {events.shape[0]:,} | frames: {len(frames)}")

    with h5py.File(os.path.join(out_dir, f"{seq}_data.hdf5"), "w") as f:
        g = f.create_group("davis/left")
        g.create_dataset("events", data=events, compression="gzip", compression_opts=1)
        g.create_dataset("image_raw", data=np.stack(frames),
                         compression="gzip", compression_opts=1)
        g.create_dataset("image_raw_ts", data=np.array(frame_ts))

    # ---------------- gt bag: poses + depth maps -----------------------------------
    poses, pose_ts, depths, depth_ts = [], [], [], []
    with Reader(gt_bag) as r:
        conns = [c for c in r.connections
                 if c.topic in ("/davis/left/pose", "/davis/left/depth_image_raw")]
        for c, t_ns, raw in r.messages(connections=conns):
            msg = ts.deserialize_ros1(raw, c.msgtype)
            stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            if c.topic == "/davis/left/pose":
                q = msg.pose.orientation
                p = msg.pose.position
                w, x, y, z = q.w, q.x, q.y, q.z
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
                M = np.eye(4); M[:3, :3] = R; M[:3, 3] = (p.x, p.y, p.z)
                poses.append(M); pose_ts.append(stamp)
            else:
                d = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width)
                depths.append(d.copy()); depth_ts.append(stamp)
    print(f"  poses: {len(poses)} | depth maps: {len(depths)}")

    with h5py.File(os.path.join(out_dir, f"{seq}_gt.hdf5"), "w") as f:
        g = f.create_group("davis/left")
        g.create_dataset("pose", data=np.stack(poses))
        g.create_dataset("pose_ts", data=np.array(pose_ts))
        g.create_dataset("depth_image_raw", data=np.stack(depths),
                         compression="gzip", compression_opts=1)
        g.create_dataset("depth_image_raw_ts", data=np.array(depth_ts))
    print(f"  wrote {seq}_data.hdf5 + {seq}_gt.hdf5 in {out_dir}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
