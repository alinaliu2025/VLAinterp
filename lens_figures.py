"""
lens_figures.py

Per-action-dimension logit-lens charts for one --mode lens run. CPU only.

    python lens_figures.py runs/run2                 # all episodes
    python lens_figures.py runs/run2 --success_only  # drop failed episodes

Averaging is hierarchical, so every task counts equally no matter how many
steps its episodes took (failed episodes run the full 220 steps):
    steps -> mean per episode -> mean per task (its 5 episodes) -> mean of tasks
The shaded band is the range (min to max) of the 10 per-task means.

Writes to <run>/plots/:
    fig_dof_agreement.png   % of steps where layer's bin == final bin
    fig_dof_distance.png    mean |layer bin - final bin|, as % of the dim's range
    fig_dof_summary.csv     the plotted numbers
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
TITLES = ["x (forward/back)", "y (left/right)", "z (up/down)", "roll", "pitch", "yaw",
          "gripper (open/close)"]
# Fixed categorical order (same as experiments.py), only for the overlay panel.
DIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
LINE, BAND = "#2a78d6", "#b7d3f6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BASE = "#898781"


def load(run_dir, success_only):
    eps = []
    for f in sorted(glob.glob(os.path.join(run_dir, "lens", "*.npz"))):
        d = np.load(f)
        if success_only and not int(d["success"]):
            continue
        eps.append({"A": d["argmax"].astype(np.int16), "task": int(d["task_id"])})
    if not eps:
        raise SystemExit(f"no lens files in {run_dir}/lens")
    return eps


def hierarchical(eps, per_step):
    """per_step(A) -> [T, 7, L] values; returns overall [7, L], per-task [n_tasks, 7, L]."""
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task"], []).append(per_step(e["A"]).mean(0))
    tasks = sorted(by_task)
    per_task = np.stack([np.mean(by_task[t], axis=0) for t in tasks])
    return per_task.mean(0), per_task


def style(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(0, 32)
    ax.set_xticks([0, 8, 16, 24, 32])
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def small_multiples(mean, per_task, baseline, title, ylabel, ylim, better, path, n_eps):
    layers = np.arange(mean.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for k, ax in enumerate(axes.flat[:7]):
        style(ax, ylabel if k % 4 == 0 else None)
        ax.fill_between(layers, per_task[:, k].min(0), per_task[:, k].max(0),
                        color=BAND, linewidth=0, label="range across the 10 tasks")
        ax.plot(layers, mean[k], color=LINE, linewidth=2, label="mean over tasks")
        ax.axhline(baseline[k], color=BASE, linestyle="--", linewidth=1.2,
                   label="always guess the most common bin")
        cross = np.flatnonzero(better(mean[k], baseline[k]))
        if len(cross):
            c = int(cross[0])
            ax.plot([c], [mean[k, c]], marker="o", markersize=6, color=LINE,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
            ax.annotate(f"beats baseline\nat layer {c}", (c, mean[k, c]),
                        xytext=(-8, 10 if ylim[1] > 50 and mean[k, c] < 60 else -26),
                        textcoords="offset points", ha="right", fontsize=7.5, color=INK)
        ax.set_title(TITLES[k], loc="left", fontsize=10, color=INK)
        ax.set_ylim(*ylim)

    # 8th panel: all seven means on one axis, for comparison.
    ax = axes.flat[7]
    style(ax)
    for k in range(7):
        ax.plot(layers, mean[k], color=DIM_COLORS[k], linewidth=1.6, label=ACTION_NAMES[k])
    ax.set_title("all seven (means)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left" if better(1, 0) else "lower left")
    for ax in axes[1]:
        ax.set_xlabel("layer (0 = input embedding, 32 = output)", color=MUTED, fontsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=3,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.935, f"{n_eps} episodes. Steps averaged within each episode, episodes "
             "within each task, then tasks averaged equally.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--success_only", action="store_true")
    args = p.parse_args()

    eps = load(args.run_dir, args.success_only)
    tag = "_success" if args.success_only else ""
    out = os.path.join(args.run_dir, "plots")
    os.makedirs(out, exist_ok=True)

    agree, agree_t = hierarchical(eps, lambda A: (A == A[..., -1:]).astype(float))
    dist, dist_t = hierarchical(eps, lambda A: np.abs(A - A[..., -1:]) / 255.0)
    agree, agree_t, dist, dist_t = agree * 100, agree_t * 100, dist * 100, dist_t * 100

    # Baselines, averaged the same hierarchical way: always output the most
    # common final bin of that dimension (computed over the selected episodes).
    final = np.concatenate([e["A"][:, :, -1] for e in eps])
    mode = np.array([np.bincount(final[:, k]).argmax() for k in range(7)])
    base_agree, _ = hierarchical(eps, lambda A: (A[:, :, -1:] == mode[None, :, None]).astype(float))
    base_dist, _ = hierarchical(eps, lambda A: np.abs(A[:, :, -1:] - mode[None, :, None]) / 255.0)
    base_agree, base_dist = base_agree[:, 0] * 100, base_dist[:, 0] * 100

    n = len(eps)
    small_multiples(
        agree, agree_t, base_agree,
        "Logit lens: how often each layer already picks the final bin",
        "% of steps matching final bin", (-2, 102), lambda m, b: m > b,
        os.path.join(out, f"fig_dof_agreement{tag}.png"), n)
    small_multiples(
        dist, dist_t, base_dist,
        "Logit lens: how far each layer's bin is from the final bin",
        "distance, % of the dim's range", (0, max(55, float(dist_t.max()) + 2)),
        lambda m, b: m < b,
        os.path.join(out, f"fig_dof_distance{tag}.png"), n)

    path = os.path.join(out, f"fig_dof_summary{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "dim", "agree_pct_mean", "agree_pct_task_min", "agree_pct_task_max",
                    "dist_pct_mean", "dist_pct_task_min", "dist_pct_task_max",
                    "baseline_agree_pct", "baseline_dist_pct"])
        for l in range(agree.shape[1]):
            for k in range(7):
                w.writerow([l, ACTION_NAMES[k], f"{agree[k, l]:.2f}", f"{agree_t[:, k, l].min():.2f}",
                            f"{agree_t[:, k, l].max():.2f}", f"{dist[k, l]:.2f}",
                            f"{dist_t[:, k, l].min():.2f}", f"{dist_t[:, k, l].max():.2f}",
                            f"{base_agree[k]:.2f}", f"{base_dist[k]:.2f}"])
    print(f"[csv]  {path}")


if __name__ == "__main__":
    main()
