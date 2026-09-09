"""
tools/blend_export_points.py -- run INSIDE Blender (headless) to export the scene's
ground-truth geometry as a point cloud (.npy) for Chamfer evaluation.

Usage:
    blender -b scene.blend --python tools/blend_export_points.py -- out.npy [n_points]

Collects all evaluated mesh objects (modifiers applied, world-transformed) and samples
up to n_points vertices uniformly (surface sampling would be better; vertex sampling is
sufficient for a scale-normalised Chamfer comparison).
"""
import sys
import numpy as np
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_path = argv[0]
n_points = int(argv[1]) if len(argv) > 1 else 100_000

dg = bpy.context.evaluated_depsgraph_get()
pts = []
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    ev = obj.evaluated_get(dg)
    try:
        mesh = ev.to_mesh()
    except RuntimeError:
        continue
    if mesh is None or len(mesh.vertices) == 0:
        ev.to_mesh_clear()
        continue
    mw = np.array(ev.matrix_world)
    v = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    v = v @ mw[:3, :3].T + mw[:3, 3]
    pts.append(v)
    ev.to_mesh_clear()

if not pts:
    raise SystemExit("no mesh vertices found")
P = np.concatenate(pts).astype(np.float32)
if P.shape[0] > n_points:
    idx = np.random.default_rng(0).choice(P.shape[0], n_points, replace=False)
    P = P[idx]
np.save(out_path, P)
print(f"exported {P.shape[0]} points -> {out_path}")
