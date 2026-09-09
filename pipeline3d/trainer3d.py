"""
trainer3d.py  --  the real 3D EvSpike-GS streaming trainer (backend-agnostic).

Reuses the SAME novel modules validated in 2D:
    evspikegs.lif_gate.LIFGate         (leaky integrate-and-fire active-set gate)
    evspikegs.event_loss               (event photometric loss + anchor)

Works with either render backend from render3d.py ('torch' on CPU, 'gsplat' on GPU).

Additions over the minimal version:
  * `budget=B`      -- hard cap on the active set per window (claim C3).
  * per-step timing -- wall-clock ms per streaming step, for the latency-vs-B experiment.
  * densification   -- clone/split/prune applied ONLY on the active set, using gradients
                       accumulated from fired Gaussians (reconciles adaptive density
                       control with the LIF gate, roadmap section 3.5).
  * `streaming_step_events` -- same update but driven by a REAL accumulated event map dE
                       (from a dataset loader) instead of synthetic GT renders.
"""
from __future__ import annotations
import os, sys, time
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evspikegs.lif_gate import LIFGate
from evspikegs.event_loss import event_photometric_loss, anchor_loss
from evspikegs.event_data import accumulate_events
from render3d import (Gaussians3D, render_torch, render_gsplat,
                      stimulus_from_projection, render_gsplat_subset,
                      render_gsplat_inactive_cache)


def _log(img, eps=1e-3):
    return torch.log(img + eps)


# densification defaults (fractions of the current population, kept deliberately gentle)
DENSIFY_GRAD_QUANTILE = 0.90     # clone/split Gaussians above this accumulated-grad quantile
SPLIT_SCALE_QUANTILE = 0.80      # ... and split (instead of clone) if scale above this
PRUNE_OPACITY = 0.005            # prune Gaussians with alpha below this
CLONE_JITTER = 0.3               # positional jitter of a clone, in units of its scale
SPLIT_SHRINK = 0.6               # scale multiplier for the two halves of a split


class EvSpikeGSTrainer3D:
    def __init__(self, gaussians, backend="torch", lr=0.02,
                 lam=0.6, theta=0.08, mode="lif", C=0.12, budget=None, bg=0.0,
                 max_scale=0.5, scene_extent=None, cull=False, anchor_weight=0.3,
                 anchor_ungated=False):
        self.g = gaussians
        self.anchor_weight = anchor_weight   # truoc day hardcode 0.3 o 2 noi ben duoi
        self.anchor_ungated = anchor_ungated   # neu True: anchor tach rieng, khong qua LIF gate
        self.backend = backend
        # cull=True: non-firing Gaussians are removed from the forward AND backward pass
        # (gsplat backend only). Realises the compute saving instead of only the update
        # saving; see streaming_step_events and render3d.render_gsplat_subset.
        self.cull = cull and backend == "gsplat"
        self.lr = lr
        self.bg = bg      # background brightness composited behind the Gaussians
        self.max_scale = max_scale   # world-space cap; ~1/5 of object radius works well
        self.scene_extent = scene_extent
        base = render_torch if backend == "torch" else render_gsplat
        self.render = lambda g, cam: base(g, cam, bg=self.bg)
        self.opt = self._make_optimizer()
        self.gate = LIFGate(gaussians.n, lam=lam, theta=theta, mode=mode, budget=budget)
        self.C = C
        self.last_step_ms = 0.0
        # accumulated |grad| of means over fired windows, for active-set densification
        self._grad_accum = torch.zeros(gaussians.n)
        self._grad_count = torch.zeros(gaussians.n)

    def _make_optimizer(self):
        """
        scene_extent=None: single-lr Adam (toy scenes).
        scene_extent set: 3DGS-style per-parameter-group learning rates — positions move
        at ~1.6e-4 x extent while opacity/brightness adapt fast. A single large lr makes
        the means wander chaotically and the reconstruction never gains structure.
        """
        if self.scene_extent is None:
            return torch.optim.Adam(self.g.params, lr=self.lr)
        ext = self.scene_extent
        groups = [
            {"params": [self.g.means], "lr": 1.6e-4 * ext},
            {"params": [self.g.log_scales], "lr": 5e-3},
            {"params": [self.g.quats], "lr": 1e-3},
            {"params": [self.g.opacity_logit], "lr": 5e-2},
            {"params": [self.g.brightness], "lr": 2.5e-2},
        ]
        return torch.optim.Adam(groups)

    def _render_log(self, cam):
        img, w, z = self.render(self.g, cam)
        return img, _log(img), w

    # ----------------------------------------------------------------------------------
    # streaming updates
    # ----------------------------------------------------------------------------------
    def streaming_step(self, cam_prev, cam_curr, gt_render_prev, gt_render_curr,
                       anchor=None, steps=2):
        """Synthetic driver: derive the event map from GT renders, then update."""
        dE = accumulate_events(_log(gt_render_prev), _log(gt_render_curr), self.C).detach()
        return self.streaming_step_events(cam_prev, cam_curr, dE.reshape(-1),
                                          anchor=anchor, steps=steps)

    def streaming_step_events(self, cam_prev, cam_curr, dE, anchor=None, steps=2,
                              pixel_mask=None):
        """
        One online update from a REAL accumulated event map dE (HW,) for the window
        [t_prev, t_curr]. This is the entry point for dataset loaders (datasets.py).
        `pixel_mask` (HW, bool) restricts the event loss to a pixel subset — e.g. the
        green Bayer sites of a color event camera when fitting a grayscale field.
        Returns (fire_frac, n_updated).
        """
        on_cuda = self.g.means.is_cuda
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_fire, last_upd = 1.0, self.g.n

        if self.cull:
            return self._culled_step(cam_prev, cam_curr, dE, anchor, steps,
                                     pixel_mask, t0, on_cuda)

        for _ in range(steps):
            self.opt.zero_grad()
            img_prev, Lp, _ = self._render_log(cam_prev)
            img_curr, Lc, w = self._render_log(cam_curr)
            dLhat = (Lc - Lp).reshape(-1)
            if pixel_mask is not None:
                loss = event_photometric_loss(dLhat[pixel_mask], dE[pixel_mask])
            else:
                loss = event_photometric_loss(dLhat, dE)
            if anchor is not None and not self.anchor_ungated:
                img_a, _, _ = self._render_log(anchor["cam"])
                loss = loss + self.anchor_weight * anchor_loss(img_a, anchor["image"])
            loss.backward()
            with torch.no_grad():
                if w is not None:
                    residual = (dLhat.detach() - dE).abs()
                    if pixel_mask is not None:
                        residual = residual * pixel_mask
                    s = self.gate.stimulus(w.detach(), residual)
                else:
                    # gsplat backend: no per-Gaussian blending weights exposed, but the
                    # backward pass has already run — the gradient norm IS the routed
                    # error signal (spec variant s_k = ||dL/d theta_k||). More faithful
                    # than point-sampling the residual at projected centers.
                    s = LIFGate.stimulus_from_grads(self.g.params)
            mask = self.gate.step(s)
            self.gate.apply_gate(self.g.params, mask)
            with torch.no_grad():
                if self.g.means.grad is not None:
                    gnorm = self.g.means.grad.norm(dim=1)
                    if self._grad_accum.device != gnorm.device:
                        self._grad_accum = self._grad_accum.to(gnorm.device)
                        self._grad_count = self._grad_count.to(gnorm.device)
                    self._grad_accum[mask] += gnorm[mask]
                    self._grad_count[mask] += 1
            frozen = LIFGate.freeze_snapshot(self.g.params)
            self.opt.step()
            LIFGate.restore_frozen(self.g.params, mask, frozen)
            self.g.clamp_(max_scale=self.max_scale)

            if anchor is not None and self.anchor_ungated:
                self.opt.zero_grad()
                img_a, _, _ = self._render_log(anchor["cam"])
                a_loss = self.anchor_weight * anchor_loss(img_a, anchor["image"])
                a_loss.backward()
                self.opt.step()               # KHONG apply_gate/restore_frozen o day
                self.g.clamp_(max_scale=self.max_scale)

            last_fire = self.gate.last_fire_frac
            last_upd = int(mask.sum().item())
        if on_cuda:
            torch.cuda.synchronize()   # honest wall-clock on GPU
        self.last_step_ms = (time.perf_counter() - t0) * 1e3
        return last_fire, last_upd

    def _culled_step(self, cam_prev, cam_curr, dE, anchor, steps, pixel_mask,
                     t0, on_cuda):
        """
        Streaming update with the non-firing Gaussians culled from the rasteriser.

        Order of operations (mirrors the controlled 2D budget experiment):
          1. ONE no-grad full render at both cameras -> event residual -> projection
             stimulus -> LIF gate picks the active set. O(N) but gradient-free.
          2. The frozen Gaussians are rendered ONCE into a detached cache per camera.
          3. Each optimisation step rasterises and differentiates ONLY the active set,
             composited over that cache, so forward+backward scale with |active|.
        The projection stimulus is used here (not gradient norms) because the gate must
        be decided BEFORE the backward pass exists, and because culled Gaussians would
        otherwise receive zero stimulus forever and could never re-enter the active set.
        """
        with torch.no_grad():
            img_p, w_p, _ = self.render(self.g, cam_prev)
            img_c, w_c, _ = self.render(self.g, cam_curr)
            dLhat0 = (_log(img_c) - _log(img_p)).reshape(-1)
            residual = (dLhat0 - dE).abs()
            if pixel_mask is not None:
                residual = residual * pixel_mask
            s = stimulus_from_projection(self.g, cam_curr, residual)
        mask = self.gate.step(s)

        idx = mask.nonzero(as_tuple=True)[0]
        cache_p = render_gsplat_inactive_cache(self.g, cam_prev, mask, bg=self.bg)
        cache_c = render_gsplat_inactive_cache(self.g, cam_curr, mask, bg=self.bg)

        for _ in range(steps):
            self.opt.zero_grad()
            ip = render_gsplat_subset(self.g, cam_prev, idx, bg_image=cache_p, bg=self.bg)
            ic = render_gsplat_subset(self.g, cam_curr, idx, bg_image=cache_c, bg=self.bg)
            dLhat = (_log(ic) - _log(ip)).reshape(-1)
            if pixel_mask is not None:
                loss = event_photometric_loss(dLhat[pixel_mask], dE[pixel_mask])
            else:
                loss = event_photometric_loss(dLhat, dE)
            if anchor is not None:
                cache_a = render_gsplat_inactive_cache(self.g, anchor["cam"], mask,
                                                       bg=self.bg)
                ia = render_gsplat_subset(self.g, anchor["cam"], idx,
                                          bg_image=cache_a, bg=self.bg)
                loss = loss + self.anchor_weight * anchor_loss(ia, anchor["image"])
            loss.backward()
            with torch.no_grad():
                if self.g.means.grad is not None:
                    gnorm = self.g.means.grad.norm(dim=1)
                    if self._grad_accum.device != gnorm.device:
                        self._grad_accum = self._grad_accum.to(gnorm.device)
                        self._grad_count = self._grad_count.to(gnorm.device)
                    self._grad_accum[mask] += gnorm[mask]
                    self._grad_count[mask] += 1
            self.gate.apply_gate(self.g.params, mask)
            frozen = LIFGate.freeze_snapshot(self.g.params)
            self.opt.step()
            LIFGate.restore_frozen(self.g.params, mask, frozen)
            self.g.clamp_(max_scale=self.max_scale)

        if on_cuda:
            torch.cuda.synchronize()
        self.last_step_ms = (time.perf_counter() - t0) * 1e3
        return self.gate.last_fire_frac, int(mask.sum().item())

    # ----------------------------------------------------------------------------------
    # densification / pruning on the ACTIVE SET (roadmap 3.5)
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def densify_and_prune(self, max_gaussians=None):
        """
        Clone/split/prune using gradients accumulated ONLY from fired Gaussians, so the two
        sparsity mechanisms do not fight each other. Rebuilds the optimizer and resizes the
        LIF membrane (survivors keep their charge, new Gaussians start at V=0).
        """
        g = self.g
        avg_grad = self._grad_accum / self._grad_count.clamp_min(1.0)
        fired = self._grad_count > 0
        alpha = torch.sigmoid(g.opacity_logit)

        keep = alpha >= PRUNE_OPACITY                                    # prune
        active_grads = avg_grad[fired]
        if active_grads.numel() == 0:
            return 0, 0, int((~keep).sum())
        thresh = torch.quantile(active_grads, DENSIFY_GRAD_QUANTILE)
        grow = fired & keep & (avg_grad >= thresh) & (avg_grad > 0)
        if max_gaussians is not None:
            room = max(0, max_gaussians - int(keep.sum()))
            if int(grow.sum()) > room:
                idx = torch.topk(torch.where(grow, avg_grad, torch.zeros_like(avg_grad)),
                                 room).indices if room > 0 else torch.empty(0, dtype=torch.long)
                grow = torch.zeros_like(grow)
                grow[idx] = True

        scale = torch.exp(g.log_scales).mean(dim=1)
        big = scale >= torch.quantile(scale, SPLIT_SCALE_QUANTILE)
        split = grow & big
        clone = grow & ~big

        keep_idx = keep.nonzero(as_tuple=True)[0]
        new_means, new_ls, new_quats, new_ol, new_b = [], [], [], [], []

        def _spawn(idx, shrink, jitter):
            for i in idx.tolist():
                sc = torch.exp(g.log_scales[i]).mean()
                new_means.append(g.means[i] + jitter * sc *
                                 torch.randn(3, device=g.means.device))
                new_ls.append(g.log_scales[i] + torch.log(torch.tensor(shrink)))
                new_quats.append(g.quats[i])
                new_ol.append(g.opacity_logit[i])
                new_b.append(g.brightness[i])

        _spawn(clone.nonzero(as_tuple=True)[0], shrink=1.0, jitter=CLONE_JITTER)
        _spawn(split.nonzero(as_tuple=True)[0], shrink=SPLIT_SHRINK, jitter=CLONE_JITTER)
        # a split replaces the parent with two smaller halves: shrink the surviving parent too
        g.log_scales.data[split] += torch.log(torch.tensor(SPLIT_SHRINK))

        n_new = len(new_means)
        means = torch.cat([g.means.data[keep_idx]] + ([torch.stack(new_means)] if n_new else []))
        ls = torch.cat([g.log_scales.data[keep_idx]] + ([torch.stack(new_ls)] if n_new else []))
        quats = torch.cat([g.quats.data[keep_idx]] + ([torch.stack(new_quats)] if n_new else []))
        ol = torch.cat([g.opacity_logit.data[keep_idx]] + ([torch.stack(new_ol)] if n_new else []))
        b = torch.cat([g.brightness.data[keep_idx]] + ([torch.stack(new_b)] if n_new else []))

        self.g = Gaussians3D(means, ls, quats, ol, b)
        self.opt = self._make_optimizer()
        self.gate.resize(keep_idx, n_new)
        self._grad_accum = torch.zeros(self.g.n)
        self._grad_count = torch.zeros(self.g.n)
        return int(clone.sum()), int(split.sum()), int((~keep).sum())