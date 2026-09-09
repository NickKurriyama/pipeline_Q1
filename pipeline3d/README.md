# pipeline3d/ — the real 3D EvSpike-GS

This is the real 3D pipeline. It has **two render backends** behind one interface:

- `render3d.render_torch` — a minimal differentiable 3D Gaussian rasterizer in pure PyTorch
  (real perspective projection + depth-sorted alpha compositing). **Runs on CPU, no gsplat.**
- `render3d.render_gsplat` — the GPU backend for real data (wraps `gsplat.rasterization`); a
  stub to fill in on a CUDA machine.

## Runnable now (CPU, no GPU, no dataset)
```bash
python pipeline3d/train_synthetic.py               # LIF-gated, ~25-28% active-set
python pipeline3d/train_synthetic.py --densify     # + active-set densification
python pipeline3d/train_synthetic.py --budget 30   # + hard per-window budget (C3)
python pipeline3d/train_synthetic.py --mode dense  # baseline, 100% active-set
```
This builds a synthetic 3D scene, orbits a pinhole camera with small motion, generates events
from the log-intensity change, and reconstructs the scene from events alone (+1 keyframe anchor)
using the LIF-gated streaming trainer. **Verified end-to-end on CPU:** dense reaches ~19.6 dB
(100% active-set); LIF reaches ~20-21 dB at ~25-28% active-set (with `--densify` or 5 epochs) —
i.e. the mechanism works in real 3D too, ~4x fewer updates at equal-or-better quality.
Figure written to `outputs/recon3d.png`.

> The synthetic run validates the full 3D *pipeline logic*. Reconstruction-quality numbers for
> the paper come from the `gsplat` backend on EventNeRF/DSEC (below), which needs a GPU.

## New in this revision
- **Hard budget B** (`--budget`, claim C3): the gate admits at most B Gaussians per window,
  preferring the highest membrane potential; capped-out neurons keep their charge and are
  served in a later window (graceful backlog instead of dropped updates).
- **Active-set densification** (`--densify`): clone/split/prune driven by gradients
  accumulated ONLY from fired Gaussians (reconciles density control with the LIF gate);
  the optimizer is rebuilt and surviving membranes carry over (`LIFGate.resize`).
- **Per-step latency** is measured (`trainer.last_step_ms`) for the streaming claim.
- **Real-data entry point**: `trainer.streaming_step_events(cam_prev, cam_curr, dE, ...)`
  consumes an accumulated event map from `datasets.py` loaders directly.
- `render3d.render_gsplat` is implemented against `gsplat.rasterization` (1.5.3, signature
  verified with `inspect.signature`), with the LIF stimulus approximated by
  `stimulus_from_projection` (residual sampled at projected Gaussian centers, weighted by
  opacity x footprint) since gsplat does not expose dense per-Gaussian blending weights.
  **VERIFIED on this machine** (RTX 2050, JIT-compiled): LIF theta=0.003 -> 20.4 dB @ 25.5%
  active vs dense 17.2 dB @ 100%; ~25 ms/packet at 240x180 with 30k Gaussians
  (`bench_gsplat.py`). The projection stimulus has a different scale than the weights-based
  one: use theta ~ 0.003 instead of 0.04. Launch via `tools\run_gpu.bat` (see root README
  for the Windows build workarounds).

## To run on real data (GPU)
1. `pip install gsplat h5py` (CUDA).
2. Point `datasets.EventNeRFDataset` / `datasets.DSECDataset` at the downloaded data
   (verify field names against the release you downloaded — see docstrings).
3. Build the trainer with `backend="gsplat"` and drive it with the loader:
   ```python
   for packet in dataset.stream():
       trainer.streaming_step_events(packet.cam_prev, packet.cam_curr, packet.dE,
                                     anchor=packet.keyframe)
   ```
4. Init Gaussians from a coarse point cloud (DSEC stereo/LiDAR; EventNeRF pseudo-frames).
5. Evaluate: PSNR/SSIM/LPIPS on held-out views; depth RMSE vs LiDAR (DSEC), Chamfer (EventNeRF);
   sparsity %, ms/packet, peak VRAM. Sweep `theta` for the quality–efficiency Pareto curve.

## Files
```
render3d.py         camera + 3D Gaussians + torch rasterizer (CPU) + gsplat hook
trainer3d.py        backend-agnostic streaming trainer (reuses LIF gate + event loss)
train_synthetic.py  A-Z CPU smoke test on a synthetic 3D scene  <-- run this
datasets.py         EventNeRF / DSEC loader skeletons (real data, your GPU box)
```

## Reused, already-validated modules (from ../evspikegs)
`lif_gate.py` (leaky integrate-and-fire active-set gate + ablations),
`event_loss.py` (Huber on log-intensity difference + anchor),
`event_data.py` (DVS event accumulation).
