"""
tools/make_paper_figures.py -- generate the paper's figures from the measured CSVs.

Every figure is built from `outputs/*.csv` (the same files the tables cite), so a
figure can never drift from the numbers in the text. Vector PDF for LaTeX + PNG preview.

Figures
  fig_pareto.pdf      quality vs active-set, controlled 2D (3 seeds) | 3D gsplat  [claim C1]
  fig_budget.pdf      per-window cost and quality vs hard budget B   [claim C3]
  fig_qualitative.pdf GT / dense / LIF renders on a held-out view    [needs --qualitative, GPU]

Design: colorblind-safe categorical pair (validated: adjacent CVD dE 9.7 deutan,
27.1 normal, contrast >3:1 on white), single y-axis per panel (never dual-axis),
recessive grid, selective direct labels, legend whenever >= 2 series.

Run:  python tools/make_paper_figures.py                 # plots from CSV (CPU)
      tools\\run_gpu.bat tools\\make_paper_figures.py --qualitative   # + renders (GPU)
"""
from __future__ import annotations
import os, sys, csv, argparse
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "paper", "figs")

# validated categorical palette (fixed order, never cycled)
C_LIF, C_DENSE, C_THIRD = "#0B6BCB", "#C4320A", "#12805C"
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#d8d8d8", "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read_csv(name: str) -> List[Dict[str, str]]:
    with open(os.path.join(ROOT, "outputs", name), newline="") as f:
        return list(csv.DictReader(f))


def style(ax) -> None:
    ax.grid(alpha=0.45, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, stem: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote paper/figs/{stem}.pdf (+ .png)")


# ------------------------------------------------------------------------------------------
def fig_pareto() -> None:
    """C1: quality vs active-set. Left = controlled 2D (3 seeds), right = 3D gsplat."""
    r2 = read_csv("pareto.csv")
    dense = [r for r in r2 if r["mode"] == "dense"][0]
    lif = [r for r in r2 if r["mode"] == "lif"]
    r3 = read_csv("pareto3d_gsplat.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.55))

    # --- left: 2D, mean +/- std over 3 seeds -----------------------------------------
    x = [float(r["active_pct_mean"]) for r in lif]
    y = [float(r["psnr_mean"]) for r in lif]
    xe = [float(r["active_pct_std"]) for r in lif]
    ye = [float(r["psnr_std"]) for r in lif]
    dp, dps = float(dense["psnr_mean"]), float(dense["psnr_std"])
    a1.axhspan(dp - dps, dp + dps, color=C_DENSE, alpha=0.10, lw=0)
    a1.axhline(dp, color=C_DENSE, ls="--", lw=1.2)
    a1.errorbar(x, y, xerr=xe, yerr=ye, fmt="o-", ms=5, lw=1.8, capsize=2.5,
                color=C_LIF, ecolor=C_LIF, elinewidth=0.9, label="EvSpike-GS (LIF)")
    a1.plot([100], [dp], marker="*", ms=11, color=C_DENSE, ls="none",
            label=r"dense ($\theta\!\to\!0$)")
    # selective direct labels, offset per point to avoid the dense line / axis edges
    off2d = {0.02: (7, 2), 0.04: (8, 6), 0.08: (7, -10), 0.40: (7, 3)}
    for r, xv, yv in zip(lif, x, y):
        th = float(r["theta"])
        if th in off2d:
            a1.annotate(rf"$\theta$={r['theta']}", (xv, yv), textcoords="offset points",
                        xytext=off2d[th], fontsize=7, color=MUTED)
    a1.set_xlabel("active set (% of Gaussians per window)")
    a1.set_ylabel("PSNR (dB)")
    a1.set_title("controlled 2D study (3 seeds)")
    a1.set_xlim(-3, 108)
    a1.legend(loc="lower right", frameon=False)
    style(a1)

    # --- right: 3D gsplat -------------------------------------------------------------
    d3 = [r for r in r3 if float(r["theta"]) == 0.0]
    l3 = [r for r in r3 if float(r["theta"]) > 0.0]
    x3 = [float(r["sparsity_pct"]) for r in l3]
    y3 = [float(r["psnr"]) for r in l3]
    if d3:
        a2.axhline(float(d3[0]["psnr"]), color=C_DENSE, ls="--", lw=1.2)
        a2.plot([100], [float(d3[0]["psnr"])], marker="*", ms=11, color=C_DENSE,
                ls="none", label=r"dense ($\theta\!=\!0$)")
    a2.plot(x3, y3, "o-", ms=5, lw=1.8, color=C_LIF, label="EvSpike-GS (LIF)")
    off3d = {0.001: (-4, -12), 0.003: (7, -2), 0.01: (7, -9), 0.03: (7, 2)}
    for r, xv, yv in zip(l3, x3, y3):
        th = float(r["theta"])
        if th in off3d:
            a2.annotate(rf"$\theta$={r['theta']}", (xv, yv), textcoords="offset points",
                        xytext=off3d[th], fontsize=7, color=MUTED)
    a2.set_xlabel("active set (% of Gaussians per window)")
    a2.set_ylabel("PSNR (dB)")
    a2.set_title("3D pipeline, gsplat CUDA rasteriser")
    a2.set_xlim(-3, 108)
    a2.legend(loc="lower right", frameon=False)
    style(a2)

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_pareto")


# ------------------------------------------------------------------------------------------
def fig_budget() -> None:
    """C3: per-window cost and quality against the hard active-set budget B."""
    rows = read_csv("budget.csv")
    labels = [r["budget"] for r in rows]
    xs = list(range(len(rows)))
    ms = [float(r["ms_per_window_mean"]) for r in rows]
    mse = [float(r["ms_per_window_std"]) for r in rows]
    ps = [float(r["psnr_mean"]) for r in rows]
    pse = [float(r["psnr_std"]) for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.45))
    a1.errorbar(xs, ms, yerr=mse, fmt="o-", ms=5, lw=1.8, capsize=2.5,
                color=C_THIRD, elinewidth=0.9)
    a1.set_xticks(xs, [r"$\infty$" if l == "inf" else l for l in labels])
    a1.set_xlabel(r"budget $B$ (max active Gaussians / window)")
    a1.set_ylabel("optimisation time (ms / window)")
    a1.set_title("per-window cost is bounded by $B$")
    style(a1)

    a2.errorbar(xs, ps, yerr=pse, fmt="s-", ms=5, lw=1.8, capsize=2.5,
                color=C_LIF, elinewidth=0.9)
    a2.set_xticks(xs, [r"$\infty$" if l == "inf" else l for l in labels])
    a2.set_xlabel(r"budget $B$")
    a2.set_ylabel("PSNR (dB)")
    a2.set_title("quality degrades gracefully")
    style(a2)

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_budget")


# ------------------------------------------------------------------------------------------
def fig_qualitative(scene: str = "chair", epochs: int = 6) -> None:
    """Train dense and LIF on one EventNeRF scene and render the same held-out view."""
    sys.path.insert(0, os.path.join(ROOT, "pipeline3d"))
    import torch
    from datasets import EventNeRFDataset
    from render3d import Gaussians3D
    from trainer3d import EvSpikeGSTrainer3D
    from train_real import estimate_background
    from eval_real_suite import load_test_views
    from evspikegs.metrics import psnr

    theta = {"chair": 1e-5, "hotdog": 1e-5, "lego": 3e-6,
             "drums": 1e-5, "mic": 3e-6}[scene]
    renders, stats = {}, {}
    ds = EventNeRFDataset(os.path.join(ROOT, "data", "data", "nerf"), scene, n_windows=300)
    views = load_test_views(os.path.join(ROOT, "data", "data", "nerf"), scene, ds)
    # THREE evenly spaced held-out views, not one: the gate's quality varies more across
    # viewpoints than dense does, so a single panel would misrepresent either method.
    # views 0 -> N/4 -> N/2 walk from the anchored viewpoint to the opposite side of the
    # orbit, which is where the two methods differ most (see the paper's discussion)
    sel = [0, len(views) // 4, len(views) // 2]

    for mode in ("dense", "lif"):
        torch.manual_seed(0)
        g = Gaussians3D.random(20_000, center=(0, 0, 0), spread=0.5,
                               scale=0.01, seed=0).to("cuda")
        bg = estimate_background(ds.anchor["image"]) if ds.anchor is not None else 0.0
        tr = EvSpikeGSTrainer3D(g, backend="gsplat", mode=mode,
                                theta=(1e-9 if mode == "dense" else theta),
                                C=ds.C, bg=bg, max_scale=0.05, scene_extent=0.5)
        mask = ds.green_mask.to("cuda")
        fires, n = [], 0
        for _ in range(epochs):
            for p in ds.stream():
                kf = p.keyframe
                if kf is not None:
                    kf = {"cam": kf["cam"], "image": kf["image"].to("cuda")}
                f, _u = tr.streaming_step_events(p.cam_prev, p.cam_curr,
                                                 p.dE.to("cuda"), anchor=kf,
                                                 steps=2, pixel_mask=mask)
                fires.append(f); n += 1
                if n % 500 == 0:
                    tr.densify_and_prune(max_gaussians=40_000)
        with torch.no_grad():
            renders[mode] = [tr.render(tr.g, views[i][0])[0].cpu() for i in sel]
        allp = [psnr(tr.render(tr.g, c)[0].cpu(), r) for c, r in views]
        stats[mode] = dict(per_view=[psnr(renders[mode][k], views[i][1])
                                     for k, i in enumerate(sel)],
                           mean=sum(allp) / len(allp),
                           active=100 * sum(fires) / len(fires))
        del tr, g
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(3, 3, figsize=(5.2, 5.4))
    cols = ["ground truth", r"dense ($\theta\!\to\!0$)", "EvSpike-GS (LIF)"]
    for r_i, i in enumerate(sel):
        imgs = [views[i][1], renders["dense"][r_i], renders["lif"][r_i]]
        labels = [None, stats["dense"]["per_view"][r_i], stats["lif"]["per_view"][r_i]]
        for c_i, (ax, img, lab) in enumerate(zip(axes[r_i], imgs, labels)):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r_i == 0:
                ax.set_title(cols[c_i], fontsize=8)
            if lab is not None:
                ax.text(0.5, -0.06, f"{lab:.1f} dB", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7, color=MUTED)
            if c_i == 0:
                ax.text(-0.04, 0.5, f"view {i}", transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=7.5, color=INK)
    fig.tight_layout(h_pad=1.4, w_pad=0.4)
    save(fig, "fig_qualitative")
    print(f"  mean over all {len(views)} held-out views: "
          f"dense {stats['dense']['mean']:.2f} dB ({stats['dense']['active']:.0f}% active) | "
          f"LIF {stats['lif']['mean']:.2f} dB ({stats['lif']['active']:.1f}% active)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qualitative", action="store_true", help="also render (needs GPU)")
    ap.add_argument("--scene", type=str, default="chair")
    a = ap.parse_args()
    print("generating paper figures from measured CSVs")
    fig_pareto()
    fig_budget()
    if a.qualitative:
        fig_qualitative(a.scene)
