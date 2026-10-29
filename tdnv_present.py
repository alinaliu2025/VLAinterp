"""
tdnv_present.py

Presentation figures for run5, on the paper's TDNV scale (sum over the
k(k-1) = 90 ordered group pairs, not the per-pair average), init 0 only
(one 10 x 10 grid = 100 inputs), raw features, no min-max bands.

Curves (CPU, from the run5 CSVs; runs anywhere):
    python tdnv_present.py curves --csv_dir runs/run5/plots --out_dir runs/run5/present

Scatter panels (CPU, needs the run5 hidden states, so run on OSC):
    python tdnv_present.py scatter --grid runs/run5/grid/generic.npz --out_dir runs/run5/present

Shuffled baseline: the run5 CSV stores the median per-pair average over 200
label shuffles; times 90 it is the median sum (no pair is ever skipped).
"""

import argparse
import csv
import os

import numpy as np

from experiments import INK, MUTED, new_axes, save_fig
from tdnv_iv_analyze import EXP_COLORS, MARKERS, PALETTE, pca2, tdnv

N_PAIRS = 90
EXP_NAMES = {"exp1_by_image": "grouped by image", "exp2_by_instruction": "grouped by instruction"}


def load_curves(csv_dir, name):
    real, shuf = {e: {} for e in EXP_NAMES}, {e: {} for e in EXP_NAMES}
    for r in csv.DictReader(open(os.path.join(csv_dir, f"tdnv_{name}.csv"))):
        if r["features"] == "raw" and r["replicate"] == "init0":
            real[r["experiment"]][int(r["layer"])] = float(r["tdnv_sum"])
    for r in csv.DictReader(open(os.path.join(csv_dir, f"tdnv_shuffled_{name}.csv"))):
        if r["features"] == "raw":
            shuf[r["experiment"]][int(r["layer"])] = float(r["tdnv_avg_shuffled_median"]) * N_PAIRS
    layers = np.array(sorted(real["exp1_by_image"]))
    return layers, {e: np.array([real[e][l] for l in layers]) for e in real}, \
        {e: np.array([shuf[e][l] for l in layers]) for e in shuf}


def curves(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.sets:
        layers, real, shuf = load_curves(args.csv_dir, name)
        fig, ax = new_axes(f"TDNV by layer ({name} instructions, 10 images x 10 instructions)",
                           "TDNV (log scale)")
        # The two shuffled curves agree within a few percent: draw one chance line
        chance = (shuf["exp1_by_image"] + shuf["exp2_by_instruction"]) / 2
        ax.plot(layers, chance, color=MUTED, linewidth=1.6, linestyle=":",
                label="random groups of 10 (chance)")
        for e, col in EXP_COLORS.items():
            ax.plot(layers, real[e], color=col, linewidth=2.2, label=EXP_NAMES[e])
            i = int(np.argmin(real[e]))
            ax.annotate(f"L{layers[i]}: {real[e][i]:.3g}", (layers[i], real[e][i]),
                        xytext=(6, -12), textcoords="offset points", fontsize=8, color=col)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        save_fig(fig, os.path.join(args.out_dir, f"tdnv_paper_{name}.png"))

        path = os.path.join(args.out_dir, f"tdnv_paper_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "exp1_by_image", "exp2_by_instruction",
                        "exp1_shuffled_median", "exp2_shuffled_median"])
            for i, l in enumerate(layers):
                w.writerow([l] + [f"{v:.4g}" for v in (real["exp1_by_image"][i],
                            real["exp2_by_instruction"][i], shuf["exp1_by_image"][i],
                            shuf["exp2_by_instruction"][i])])
        print(f"[csv]  {path}")


def scatter(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(args.out_dir, exist_ok=True)
    D = np.load(args.grid)
    hidden, src = D["hidden"], D["image_src"]                 # [K, M, L+1, d], [M, 2]
    instructions = [str(s) for s in D["instructions"]]
    layouts = sorted(set(int(t) for t in src[:, 0]))
    cols = [int(np.where((src[:, 0] == t) & (src[:, 1] == args.init))[0][0]) for t in layouts]
    K, N = hidden.shape[0], len(cols)
    name = os.path.splitext(os.path.basename(args.grid))[0]
    panels = [(int(p.split(":")[0]), p.split(":")[1]) for p in args.panels.split(",")]

    fig_all, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2))
    for ax_all, (l, by) in zip(np.atleast_1d(axes), panels):
        X = hidden[:, cols, l].reshape(K * N, -1).astype(np.float64)    # raw, as in the curves
        G = X.reshape(K, N, -1)                                         # [instruction, image, d]
        exp = "exp1_by_image" if by == "image" else "exp2_by_instruction"
        val = tdnv(G.transpose(1, 0, 2) if by == "image" else G)[0]
        gidx = np.tile(np.arange(N), K) if by == "image" else np.repeat(np.arange(K), N)
        labels = [f"layout {t}" for t in layouts] if by == "image" else instructions
        P, ev = pca2(X)

        fig_one, ax_one = plt.subplots(figsize=(5.6, 7.0))
        for ax in (ax_all, ax_one):
            for g in range(len(labels)):
                pts, col = P[gidx == g], PALETTE[g % len(PALETTE)]
                hollow = g >= len(PALETTE)
                ax.scatter(pts[:, 0], pts[:, 1], s=46, marker=MARKERS[g],
                           facecolors="none" if hollow else col, edgecolors=col,
                           linewidths=1.5 if hollow else 0.6, alpha=0.9)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"layer {l}, colored by {by}\nTDNV {val:.4g}   "
                         f"(PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance)",
                         fontsize=11, color=INK, loc="left")
            handles = [Line2D([], [], linestyle="", marker=MARKERS[g], markersize=7,
                              markerfacecolor="none" if g >= len(PALETTE) else PALETTE[g % len(PALETTE)],
                              markeredgecolor=PALETTE[g % len(PALETTE)], label=lab)
                       for g, lab in enumerate(labels)]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                      ncol=1 if by == "instruction" else 2, fontsize=8, frameon=False,
                      labelcolor=MUTED)
        save_fig(fig_one, os.path.join(args.out_dir, f"scatter_{name}_L{l}_{by}.png"))
        print(f"[info] L{l} by {by}: TDNV {val:.4g} ({EXP_NAMES[exp]})")
    save_fig(fig_all, os.path.join(args.out_dir, f"scatter_{name}_panels.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--csv_dir", required=True)
    c.add_argument("--out_dir", required=True)
    c.add_argument("--sets", nargs="*", default=["generic", "suite"])
    s = sub.add_parser("scatter")
    s.add_argument("--grid", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--init", type=int, default=0)
    s.add_argument("--panels", default="3:image,7:instruction,31:image",
                   help="comma list of layer:image|instruction")
    args = p.parse_args()
    curves(args) if args.cmd == "curves" else scatter(args)


if __name__ == "__main__":
    main()
