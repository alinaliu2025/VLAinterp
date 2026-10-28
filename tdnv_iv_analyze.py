"""
tdnv_iv_analyze.py

TDNV image vs. instruction, step 2 (CPU, no model): metrics, plots, results.md
from the grids saved by tdnv_iv_extract.py.

For each instruction set and each replicate r (one init index):
    images  = the k layouts at init r
    grid    = every instruction x those images      [K, k, L+1, d]
    Exp 1   = grid grouped by IMAGE        (k groups of N instructions)
    Exp 2   = grid grouped by INSTRUCTION  (k groups of N images)
Both experiments use the same hidden states, so k and N match by construction.

TDNV (spec, sum over ordered pairs g != g'):
    TDNV = sum (var_g + var_g') / (2 ||mu_g - mu_g'||^2),  tdnv_avg = TDNV / (k(k-1))
Computed on raw features and on the lab's preprocessing (subtract the mean of
the replicate's samples, scale each to unit norm; same as instruction_swap.py).
Readout 0 (embeddings) is identical for every sample, because every prompt ends
in the same token, so it is checked and skipped.

    python tdnv_iv_analyze.py --config runs/run5/config.json --out_dir runs/run5
"""

import argparse
import csv
import json
import os

import numpy as np

from experiments import GRID, INK, MUTED, SanityCheckError, new_axes, save_fig

# Reference categorical palette (dataviz skill), fixed order. A scatter has 10
# groups, more than the 8 hues, so groups 9-10 reuse hues 1-2 with hollow
# markers, and every group also gets its own marker and a centroid label.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "o", "s"]
EXP_COLORS = {"exp1_by_image": "#2a78d6", "exp2_by_instruction": "#eb6834"}
EXP_NAMES = {"exp1_by_image": "Exp 1: grouped by image",
             "exp2_by_instruction": "Exp 2: grouped by instruction"}


def tdnv(H):
    """H: [k, N, d] float64. Returns tdnv_sum, tdnv_avg, mean_var, mean_dist2, n_skipped."""
    k = H.shape[0]
    mu = H.mean(1)
    var = ((H - mu[:, None]) ** 2).sum(-1).mean(1)
    d2 = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    off = ~np.eye(k, dtype=bool)
    ok = off & (d2 > 0)
    n_skip = int(off.sum() - ok.sum())
    ratio = (var[:, None] + var[None]) / (2 * np.where(ok, d2, 1.0))
    total = ratio[ok].sum()
    avg = total / ok.sum() if ok.any() else np.nan
    return total, avg, var.mean(), d2[off].mean(), n_skip


def preprocess(X, mode):
    """X: [n, d]. 'raw' or 'centered_unitnorm' (the lab's TDNV preprocessing)."""
    if mode == "raw":
        return X
    X = X - X.mean(0, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def shuffled_tdnv(X, k, N, rng, n):
    """Median tdnv_avg after randomly reassigning the k*N samples to k groups of N."""
    vals = [tdnv(X[rng.permutation(len(X))].reshape(k, N, -1))[1] for _ in range(n)]
    return float(np.median(vals))


def first_crossing(a, b, layers):
    s = np.sign(a - b)
    for i in range(1, len(s)):
        if s[i] != s[i - 1] and s[i] != 0:
            return int(layers[i - 1]), int(layers[i])
    return None


def pca2(X):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    return U[:, :2] * S[:2], ev[:2]


def scatter_figure(Xl_by_layer, layers, inst_labels, img_labels, tdnv_by, title, path):
    """
    One PCA per layer (columns) of the replicate-0 grid [K*M, d]; the SAME
    points are colored by image (top row, Exp 1) and by instruction (bottom row,
    Exp 2).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K, M = len(inst_labels), len(img_labels)
    inst_idx = np.repeat(np.arange(K), M)
    img_idx = np.tile(np.arange(M), K)
    ncol = len(layers)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol + 1.6, 6.2), squeeze=False)
    for c, l in enumerate(layers):
        P, ev = pca2(Xl_by_layer[l])
        for r, (gidx, n_g, exp) in enumerate([(img_idx, M, "exp1_by_image"),
                                              (inst_idx, K, "exp2_by_instruction")]):
            ax = axes[r, c]
            for g in range(n_g):
                pts = P[gidx == g]
                col = PALETTE[g % len(PALETTE)]
                hollow = g >= len(PALETTE)
                ax.scatter(pts[:, 0], pts[:, 1], s=22, marker=MARKERS[g],
                           facecolors="none" if hollow else col, edgecolors=col,
                           linewidths=1.2 if hollow else 0.6, alpha=0.9)
                cx, cy = pts.mean(0)
                ax.annotate(str(g), (cx, cy), fontsize=7, color=INK, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.8))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(f"layer {l}\nPC1 {ev[0]:.0%}, PC2 {ev[1]:.0%}", fontsize=9, color=INK)
            ax.text(0.02, 0.02, f"TDNV {tdnv_by[exp][l]:.3g}", transform=ax.transAxes,
                    fontsize=7, color=MUTED)
        if c == 0:
            axes[0, 0].set_ylabel("color = image\n(Exp 1 groups)", color=INK, fontsize=9)
            axes[1, 0].set_ylabel("color = instruction\n(Exp 2 groups)", color=INK, fontsize=9)
    from matplotlib.lines import Line2D

    def handles(labels):
        return [Line2D([], [], linestyle="", marker=MARKERS[g], markersize=6,
                       markerfacecolor="none" if g >= len(PALETTE) else PALETTE[g % len(PALETTE)],
                       markeredgecolor=PALETTE[g % len(PALETTE)], label=f"{g}: {lab}")
                for g, lab in enumerate(labels)]
    fig.legend(handles=handles(img_labels), loc="upper left", bbox_to_anchor=(1.0, 0.97),
               fontsize=7, frameon=False, title="images", title_fontsize=8)
    fig.legend(handles=handles(inst_labels), loc="upper left", bbox_to_anchor=(1.0, 0.48),
               fontsize=7, frameon=False, title="instructions", title_fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def contact_sheet(images, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(images)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.7 * nrow), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(labels[i], fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def analyze_set(name, D, cfg, out_dir, rng):
    hidden = D["hidden"]                                   # [K, M, L+1, d]
    src = D["image_src"]                                   # [M, 2] (task, init)
    instructions = [str(s) for s in D["instructions"]]
    K, M, n_read, d = hidden.shape
    k, N = cfg["k"], cfg["N"]
    layouts, inits = cfg["layout_tasks"], cfg["inits"]
    if not (K == N == k == len(layouts)):
        raise SanityCheckError(f"crossed design needs K = N = k = #layouts; got K={K}, "
                               f"k={k}, N={N}, layouts={len(layouts)}")
    if not np.isfinite(hidden).all():
        raise SanityCheckError("non-finite hidden states")
    if np.ptp(hidden[:, :, 0], axis=(0, 1)).max() != 0:
        print("[warn] readout 0 differs between samples (expected identical: same last token)")
    else:
        print("[check] readout 0 (embeddings) identical for all samples, as expected; skipped")
    layers = np.arange(1, n_read)

    rows, curves = [], {}
    for feat in ("raw", "centered_unitnorm"):
        res = {e: np.full((len(inits), n_read, 5), np.nan)
               for e in ("exp1_by_image", "exp2_by_instruction")}
        shuf = {e: np.full((len(inits), n_read), np.nan) for e in res}
        for ri, init in enumerate(inits):
            cols = [int(np.where((src[:, 0] == t) & (src[:, 1] == init))[0][0]) for t in layouts]
            for l in layers:
                X = preprocess(hidden[:, cols, l].reshape(K * N, d).astype(np.float64), feat)
                G = X.reshape(K, N, d)                    # [instruction, image, d]
                for e, H in (("exp1_by_image", G.transpose(1, 0, 2)),
                             ("exp2_by_instruction", G)):
                    out = tdnv(H)
                    if out[4]:
                        print(f"[warn] {name}/{feat}/init{init}/L{l}/{e}: "
                              f"{out[4]} pairs with zero mean distance skipped")
                    res[e][ri, l] = out
                    if not np.all(np.isfinite(out[:4])):
                        raise SanityCheckError(f"non-finite TDNV {name}/{feat}/L{l}/{e}")
                    if ri == 0:
                        shuf[e][ri, l] = shuffled_tdnv(X, k, N, rng, cfg["n_shuffles"])
                    rows.append([name, feat, e, f"init{init}", l] + [f"{v:.6g}" for v in out[:4]])
        for e in res:
            m = np.nanmean(res[e], axis=0)
            for l in layers:
                rows.append([name, feat, e, "mean", l] + [f"{v:.6g}" for v in m[l, :4]])
        curves[feat] = (res, shuf)

    # CSV: the spec's columns, plus set / features / replicate
    path = os.path.join(out_dir, f"tdnv_{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instruction_set", "features", "experiment", "replicate", "layer",
                    "tdnv_sum", "tdnv_avg", "mean_var", "mean_dist2"])
        w.writerows(rows)
    print(f"[csv]  {path}")
    path = os.path.join(out_dir, f"tdnv_shuffled_{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instruction_set", "features", "experiment", "layer",
                    "tdnv_avg_real_init0", "tdnv_avg_shuffled_median", "shuffled_over_real"])
        for feat, (res, shuf) in curves.items():
            for e in res:
                for l in layers:
                    real, sh = res[e][0, l, 1], shuf[e][0, l]
                    w.writerow([name, feat, e, l, f"{real:.6g}", f"{sh:.6g}", f"{sh / real:.4g}"])
    print(f"[csv]  {path}")

    # Main plot + decomposition, per feature mode
    summary = {}
    for feat, (res, shuf) in curves.items():
        tag = f"{name}_{feat}"
        fig, ax = new_axes(f"TDNV by layer: image vs instruction ({name} instructions, {feat})",
                           "TDNV, per-pair average (log scale)")
        for e, col in EXP_COLORS.items():
            v = res[e][:, layers, 1]
            ax.fill_between(layers, v.min(0), v.max(0), color=col, alpha=0.15, linewidth=0)
            ax.plot(layers, v.mean(0), color=col, linewidth=2, label=f"{EXP_NAMES[e]} (mean of {len(inits)} inits, band = min-max)")
            ax.plot(layers, shuf[e][0, layers], color=col, linewidth=1.2, linestyle=":",
                    label=f"{EXP_NAMES[e]}, labels shuffled (init 0)")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7)
        save_fig(fig, os.path.join(out_dir, f"tdnv_main_{tag}.png"))

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, idx, lab in ((axes[0], 2, "mean within-group variance"),
                             (axes[1], 3, "mean between-group squared distance")):
            for e, col in EXP_COLORS.items():
                v = res[e][:, layers, idx]
                ax.fill_between(layers, v.min(0), v.max(0), color=col, alpha=0.15, linewidth=0)
                ax.plot(layers, v.mean(0), color=col, linewidth=2, label=EXP_NAMES[e])
            ax.set_title(lab, loc="left", fontsize=11, color=INK)
            ax.set_xlabel("layer", color=MUTED)
            ax.set_yscale("log")
            ax.grid(True, color=GRID, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"TDNV decomposition ({name} instructions, {feat})", x=0.01, ha="left",
                     color=INK)
        save_fig(fig, os.path.join(out_dir, f"tdnv_decomp_{tag}.png"))

        m1 = res["exp1_by_image"][:, layers, 1].mean(0)
        m2 = res["exp2_by_instruction"][:, layers, 1].mean(0)
        summary[feat] = {
            "exp1_min": (int(layers[np.argmin(m1)]), float(m1.min())),
            "exp2_min": (int(layers[np.argmin(m2)]), float(m2.min())),
            "exp1_last": float(m1[-1]), "exp2_last": float(m2[-1]),
            "crossing": first_crossing(m1, m2, layers),
            "lower_everywhere": ("exp1" if (m1 < m2).all() else "exp2" if (m2 < m1).all() else None),
            # shuffled / real at each experiment's own TDNV minimum (init 0)
            "shuffle_ratio_at_min": {e: float(shuf[e][0, l] / res[e][0, l, 1]) for e, l in
                                     (("exp1_by_image", int(layers[np.argmin(m1)])),
                                      ("exp2_by_instruction", int(layers[np.argmin(m2)])))},
        }

    # Scatter: replicate 0, centered + unit-norm features (what TDNV is computed on)
    cols0 = [int(np.where((src[:, 0] == t) & (src[:, 1] == inits[0]))[0][0]) for t in layouts]
    Xl = {l: preprocess(hidden[:, cols0, l].reshape(K * N, d).astype(np.float64),
                        "centered_unitnorm") for l in cfg["scatter_layers"]}
    res = curves["centered_unitnorm"][0]
    tdnv_by = {e: res[e][0, :, 1] for e in res}
    img_labels = [f"layout {t}" for t in layouts]
    inst_labels = [s if len(s) <= 48 else s[:45] + "..." for s in instructions]
    scatter_figure(Xl, cfg["scatter_layers"], inst_labels, img_labels, tdnv_by,
                   f"PCA of last-prompt-token hidden states, {name} instructions x 10 layouts "
                   f"(init {inits[0]}, centered + unit-norm)",
                   os.path.join(out_dir, f"tdnv_scatter_{name}.png"))
    return summary, instructions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 7))

    lines = ["# run5: TDNV, image vs. instruction", "",
             f"- model: `{cfg['model_id']}`", f"- k = {cfg['k']}, N = {cfg['N']} (crossed grid: "
             f"Exp 1 and Exp 2 group the same hidden states)",
             f"- images: first frame of libero_spatial layout tasks {cfg['layout_tasks']}, "
             f"replicates = init states {cfg['inits']}",
             f"- variation: {cfg['variation']}; seed {cfg.get('seed', 7)}; "
             f"{cfg['n_shuffles']} label shuffles (init {cfg['inits'][0]})",
             "- hidden state: last prompt position (produces the first action token); "
             "readout 0 skipped (identical for all samples)", ""]
    for name in cfg["instruction_sets"]:
        path = os.path.join(args.out_dir, "grid", f"{name}.npz")
        if not os.path.isfile(path):
            print(f"[skip] {path} not found")
            continue
        D = np.load(path)
        if name == list(cfg["instruction_sets"])[0]:
            # every image exactly as the model saw it: one PNG each + a sheet per init
            import matplotlib.pyplot as plt
            src = D["image_src"]
            img_dir = os.path.join(plots, "images")
            os.makedirs(img_dir, exist_ok=True)
            for i, (t, init) in enumerate(src):
                plt.imsave(os.path.join(img_dir, f"layout{t}_init{init}.png"), D["images"][i])
            print(f"[img]  {len(src)} images in {img_dir}")
            for init in cfg["inits"]:
                sel = [i for i in range(len(src)) if src[i, 1] == init]
                contact_sheet(D["images"][sel], [f"layout {src[i, 0]} (init {init})" for i in sel],
                              os.path.join(plots, f"images_init{init}.png"))
        summary, instructions = analyze_set(name, D, cfg, plots, rng)
        lines += [f"## Instruction set `{name}`", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(instructions)] + [""]
        lines += ["| features | Exp 1 (by image) min | Exp 2 (by instruction) min | "
                  "Exp 1 / Exp 2 at last readout | curves cross | shuffled / real (min over layers) |",
                  "|---|---|---|---|---|---|"]
        for feat, s in summary.items():
            cross = (f"between layers {s['crossing'][0]} and {s['crossing'][1]}"
                     if s["crossing"] else f"no ({s['lower_everywhere']} lower at every layer)")
            sr = s["shuffle_ratio_at_min"]
            lines.append(f"| {feat} | {s['exp1_min'][1]:.3g} @ L{s['exp1_min'][0]} | "
                         f"{s['exp2_min'][1]:.3g} @ L{s['exp2_min'][0]} | "
                         f"{s['exp1_last']:.3g} / {s['exp2_last']:.3g} | {cross} | "
                         f"Exp 1 {sr['exp1_by_image']:.2g}x, Exp 2 {sr['exp2_by_instruction']:.2g}x |")
        lines.append("")
    out = os.path.join(args.out_dir, "results.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[md]   {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
