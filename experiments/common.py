"""Shared config + helpers for the mechanism experiments (2D toy, CPU).

All three claim experiments (C1 Pareto, C2 ablation, C3 budget) run on the same small
scene so numbers are comparable across scripts. Multi-seed, mean +/- std, CSV output —
the statistical protocol the roadmap requires (section 5.4), executed on the toy.
"""
from __future__ import annotations
import os, sys, csv, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# one shared scene/config for every mechanism experiment
CFG = dict(H=36, W=36, n_gauss=150, n_windows=34, steps_per_window=3,
           C=0.12, lam=0.6, theta=0.08, epochs=3)
SEEDS = [0, 1, 2]
OUTDIR = os.path.join(ROOT, "outputs")


def mean_std(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def smoothed_psnr(run, last=10):
    """Mean PSNR over the last `last` windows — less noisy than the single final value."""
    hist = run["psnr_hist"]
    tail = hist[-last:] if len(hist) >= last else hist
    return sum(tail) / len(tail)


def fmt(m, s, nd=2):
    # ASCII +/- so prints survive Windows cp1252 consoles; figures may use unicode
    return f"{m:.{nd}f} +/- {s:.{nd}f}"


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
