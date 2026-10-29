"""
tdnv_present.py

Presentation figures for run5, on the paper's TDNV scale (sum over the
k(k-1) = 90 ordered group pairs, not the per-pair average), init 0 only
(one 10 x 10 grid = 100 inputs), raw features, no min-max bands.

Curves (CPU, from the run5 CSVs; runs anywhere):
    python tdnv_present.py curves --csv_dir runs/run5/plots --out_dir runs/run5/present

Scatter panels (CPU, needs the run5 hidden states, so run on OSC):
    python tdnv_present.py scatter --grid runs/run5/grid/generic.npz --out_dir runs/run5/present

Instruction groupings at one layer (CPU, needs the run5 hidden states, so run on OSC):
    python tdnv_present.py groups --grid runs/run5/grid/generic.npz --out_dir runs/run5/present

Shuffled baseline: the run5 CSV stores the median per-pair average over 200
label shuffles; times 90 it is the median sum (no pair is ever skipped).

`groups` measures which instructions sit close together at --layer in the full
space, using the per-pair TDNV term (var_t + var_t') / (2 ||mu_t - mu_t'||^2)
(TDNV is the sum of these; high = the pair is hard to separate). Per init:
  - keyword groups (--groups): mean log10 pair term inside each group, with an
    exact p over every same-size subset of instructions; plus all keyword-sharing
    pairs vs the rest, with labels permuted --n_perm times
  - word overlap: Spearman correlation between pair terms and the instructions'
    content-word Jaccard overlap (Mantel test, labels permuted), at --layer and
    at every layer
  - --line a,b,c: is b's center on the segment from a's to c's?
  - stability: Spearman correlation of each init's pair terms with init 0's
Figures (init 0): heatmap ordered by --order, dendrogram, word-overlap curve.
"""

import argparse
import csv
import os
from itertools import combinations

import numpy as np

from experiments import INK, MUTED, SanityCheckError, new_axes, save_fig
from tdnv_iv_analyze import EXP_COLORS, MARKERS, PALETTE, pca2, tdnv

N_PAIRS = 90
EXP_NAMES = {"exp1_by_image": "grouped by image", "exp2_by_instruction": "grouped by instruction"}
# Generic set: keyword groups from the L7 scatter, and an order that makes each a contiguous block
GENERIC_GROUPS = "stove:2,6,8;drawer:3,4,5;bowl:0,1,5;plate:0,6,7,9"
GENERIC_ORDER = "2,8,6,9,7,0,1,5,3,4"
STOP = {"the", "a", "an", "and", "it", "of", "to", "on", "in"}
SEQ_BLUE = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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
        fig, ax = new_axes(f"TDNV by layer ({name} instructions, 10 images x 10 instructions)"
                           + (f"\n{args.label}" if args.label else ""), "TDNV (log scale)")
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
            ax.set_title((f"{args.label}\n" if args.label else "") +
                         f"layer {l}, colored by {by}\nTDNV {val:.4g}   "
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


def grid_by_init(D, init):
    """[K, N, L+1, d]: every instruction x the layouts at one init, layouts in task order."""
    src = D["image_src"]
    layouts = sorted(set(int(t) for t in src[:, 0]))
    cols = [int(np.where((src[:, 0] == t) & (src[:, 1] == init))[0][0]) for t in layouts]
    return D["hidden"][:, cols]


def pair_matrix(G):
    """G: [k, N, d]. Per-pair TDNV terms [k, k] (diagonal nan) and the group centers."""
    mu = G.mean(1)
    var = ((G - mu[:, None]) ** 2).sum(-1).mean(1)
    d2 = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.nan)
    return (var[:, None] + var[None]) / (2 * d2), mu


def upper(M):
    return M[np.triu_indices(len(M), 1)]


def word_overlap(instructions):
    """Jaccard overlap of content words (stopwords dropped, plural s stripped)."""
    def words(s):
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.lower().split()} - STOP
    W = [words(s) for s in instructions]
    J = np.array([[len(a & b) / len(a | b) for b in W] for a in W])
    np.fill_diagonal(J, np.nan)
    return J


def mantel(A, B, rng, n):
    """One-sided Spearman Mantel test: A's instruction labels permuted n times."""
    from scipy.stats import spearmanr
    obs = spearmanr(upper(A), upper(B))[0]
    null = np.array([spearmanr(upper(A[np.ix_(p, p)]), upper(B))[0]
                     for p in (rng.permutation(len(A)) for _ in range(n))])
    return obs, (1 + (null >= obs).sum()) / (1 + n)


def group_test(L, members):
    """Mean of L inside the group; exact p = share of same-size subsets scoring at least as high."""
    def within(S):
        return np.mean([L[a, b] for a, b in combinations(S, 2)])
    obs = within(members)
    null = np.array([within(S) for S in combinations(range(len(L)), len(members))])
    return obs, float((null >= obs - 1e-12).mean())


def shared_test(L, share, rng, n):
    """Mean of L over keyword-sharing pairs minus the rest; labels permuted n times."""
    s = upper(share).astype(bool)

    def stat(M):
        v = upper(M)
        return v[s].mean() - v[~s].mean()
    obs = stat(L)
    null = np.array([stat(L[np.ix_(p, p)]) for p in (rng.permutation(len(L)) for _ in range(n))])
    return obs, (1 + (null >= obs).sum()) / (1 + n)


def line_check(mu, a, b, c):
    """
    Is b's center on the segment a -> c? Returns the angle between b-a and c-b
    (0 = straight line), where b projects on a -> c (0 = at a, 1 = at c), and
    b's distance off that line as a fraction of |c - a|.
    """
    u, v, w = mu[b] - mu[a], mu[c] - mu[b], mu[c] - mu[a]
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    t = (u @ w) / (w @ w)
    return np.degrees(np.arccos(np.clip(cos, -1, 1))), t, np.linalg.norm(u - t * w) / np.linalg.norm(w)


def groups(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    D = np.load(args.grid)
    instructions = [str(s) for s in D["instructions"]]
    K, l = len(instructions), args.layer
    inits = sorted(set(int(i) for i in D["image_src"][:, 1]))
    name = os.path.splitext(os.path.basename(args.grid))[0]
    tag = f"{name}_L{l}"
    title_extra = f"\n{args.label}" if args.label else ""
    kw = {g.split(":")[0]: [int(i) for i in g.split(":")[1].split(",")] for g in args.groups.split(";")}
    order = [int(i) for i in args.order.split(",")]
    if sorted(order) != list(range(K)):
        raise SystemExit(f"[error] --order must list each of the {K} instructions once")
    share = np.zeros((K, K))
    for m in kw.values():
        share[np.ix_(m, m)] = 1
    np.fill_diagonal(share, np.nan)
    J = word_overlap(instructions)

    P, mus, sweep = {}, {}, {}
    for i in inits:
        H = grid_by_init(D, i).astype(np.float64)                    # [K, N, L+1, d]
        P[i], mus[i] = pair_matrix(H[:, :, l])
        if not np.isclose(np.nansum(P[i]), tdnv(H[:, :, l])[0], rtol=1e-9):
            raise SanityCheckError(f"pair terms do not sum to TDNV (init {i}, layer {l})")
        sweep[i] = np.array([spearmanr(upper(np.log10(pair_matrix(H[:, :, r])[0])), upper(J))[0]
                             for r in range(1, H.shape[2])])
    i0 = inits[0]
    print(f"[check] pair terms sum to TDNV at layer {l} for inits {inits}; "
          f"init {i0} TDNV = {np.nansum(P[i0]):.4g}")

    # Tests, per init
    rows = []
    line = [int(i) for i in args.line.split(",")] if args.line else None
    for i in inits:
        Lg = np.log10(P[i])
        for g, m in kw.items():
            obs, p = group_test(Lg, m)
            rows.append([i, f"group {g} {m}", "mean log10 pair term inside", obs, p])
        obs, p = shared_test(Lg, share, rng, args.n_perm)
        rows.append([i, "keyword-sharing pairs vs rest", "difference of mean log10 pair term", obs, p])
        obs, p = mantel(Lg, J, rng, args.n_perm)
        rows.append([i, "word overlap (Mantel)", "Spearman rho", obs, p])
        if line:
            ang, t, off = line_check(mus[i], *line)
            rows.append([i, f"line {line}", "angle (deg) / position on a->c / offset", ang, np.nan])
            rows[-1][3] = f"{ang:.1f} / {t:.2f} / {off:.2f}"
        if i != i0:
            rows.append([i, f"stability vs init {i0}", "Spearman rho of pair terms",
                         spearmanr(upper(P[i]), upper(P[i0]))[0], np.nan])
    path = os.path.join(args.out_dir, f"groups_{tag}_tests.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init", "test", "statistic", "value", "p"])
        w.writerows([r[:3] + [r[3] if isinstance(r[3], str) else f"{r[3]:.4g}",
                              "" if np.isnan(r[4]) else f"{r[4]:.4g}"] for r in rows])
    print(f"[csv]  {path}")
    print(f"\n[init {i0}, layer {l}]")
    for r in rows:
        if r[0] == i0:
            v = r[3] if isinstance(r[3], str) else f"{r[3]:.3f}"
            print(f"  {r[1]:<38s} {r[2]:<40s} {v:>16s}  " + ("" if np.isnan(r[4]) else f"p = {r[4]:.3g}"))
    for test in sorted({r[1] for r in rows if r[0] != i0 and not np.isnan(r[4])} |
                       {r[1] for r in rows if r[1].startswith("stability")}):
        vals = [r for r in rows if r[1] == test and r[0] != i0]
        if vals:
            print(f"  other inits, {test}: " + ", ".join(
                (f"{r[3]:.2f}" if not isinstance(r[3], str) else r[3]) +
                ("" if np.isnan(r[4]) else f" (p {r[4]:.2g})") for r in vals))

    # Pair table, init 0: which pairs are closer than their word overlap predicts
    iu = np.triu_indices(K, 1)
    Lg0 = np.log10(P[i0])
    fit = np.polyfit(J[iu], Lg0[iu], 1)
    resid = Lg0 - np.polyval(fit, J)
    path = os.path.join(args.out_dir, f"groups_{tag}_pairs.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "instruction_a", "instruction_b", f"pair_term_init{i0}",
                    "pair_term_geomean_inits", "word_overlap", "shares_keyword_group",
                    "log10_residual_vs_word_overlap"])
        for a, b in sorted(zip(*iu), key=lambda ab: -resid[ab]):
            gm = 10 ** np.mean([np.log10(P[i][a, b]) for i in inits])
            w.writerow([a, b, instructions[a], instructions[b], f"{P[i0][a, b]:.4g}", f"{gm:.4g}",
                        f"{J[a, b]:.3f}", int(share[a, b]), f"{resid[a, b]:+.3f}"])
    print(f"[csv]  {path}")

    # Heatmap, init 0, in --order; keyword groups outlined where they are contiguous
    M = Lg0[np.ix_(order, order)]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(K), [str(i) for i in order], color=MUTED, fontsize=9)
    ax.set_yticks(range(K), [f"{i}: {instructions[i]}" for i in order], color=INK, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    pos = {t: j for j, t in enumerate(order)}
    for g, m in kw.items():
        js = sorted(pos[t] for t in m)
        if js[-1] - js[0] + 1 != len(js):
            print(f"[warn] group {g} is not contiguous in --order; not outlined")
            continue
        ax.add_patch(Rectangle((js[0] - 0.5, js[0] - 0.5), len(js), len(js),
                               fill=False, edgecolor=INK, linewidth=1.6))
        ax.text(js[-1] + 0.55, js[0] - 0.35, g, color=INK, fontsize=9, va="top")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.09)
    cb.set_label("log10 pair TDNV term (darker = harder to separate)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"Layer {l}: pair TDNV terms between instructions (init {i0}, "
                 f"TDNV = {np.nansum(P[i0]):.3g}){title_extra}", loc="left", fontsize=11, color=INK)
    path = os.path.join(args.out_dir, f"groups_{tag}_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")       # long row labels: save_fig would clip them
    plt.close(fig)
    print(f"[plot] {path}")

    # Dendrogram, init 0: separation = 1 / sqrt(pair term)
    S = 1 / np.sqrt(P[i0])
    np.fill_diagonal(S, 0)
    Z = linkage(squareform((S + S.T) / 2, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    dendrogram(Z, labels=[f"{i}: {s}" for i, s in enumerate(instructions)], orientation="right",
               link_color_func=lambda k: INK, ax=ax, leaf_font_size=9)
    ax.set_xlabel("separation, 1 / sqrt(pair TDNV term) (average linkage)", color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.tick_params(axis="y", colors=INK, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Layer {l}: which instructions merge first (init {i0}){title_extra}",
                 loc="left", fontsize=11, color=INK)
    save_fig(fig, os.path.join(args.out_dir, f"groups_{tag}_dendrogram.png"))

    # Word overlap vs pair terms, every layer
    layers = np.arange(1, len(sweep[i0]) + 1)
    fig, ax = new_axes(f"Do instructions that share words sit closer? ({name}){title_extra}",
                       "Spearman rho, word overlap vs pair TDNV")
    for j, i in enumerate(inits[1:]):
        ax.plot(layers, sweep[i], color=MUTED, linewidth=1, alpha=0.6,
                label=f"inits {inits[1]}-{inits[-1]}" if j == 0 else None)
    ax.plot(layers, sweep[i0], color=EXP_COLORS["exp2_by_instruction"], linewidth=2.2,
            label=f"init {i0}")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(l, color=MUTED, linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, os.path.join(args.out_dir, f"groups_{name}_wordoverlap_by_layer.png"))
    path = os.path.join(args.out_dir, f"groups_{name}_wordoverlap_by_layer.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [f"init{i}" for i in inits])
        for r, lay in enumerate(layers):
            w.writerow([lay] + [f"{sweep[i][r]:.4f}" for i in inits])
    print(f"[csv]  {path}")


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
    g = sub.add_parser("groups")
    g.add_argument("--grid", required=True)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--layer", type=int, default=7)
    g.add_argument("--groups", default=GENERIC_GROUPS, help="name:i,j,...;name:... (may overlap)")
    g.add_argument("--order", default=GENERIC_ORDER, help="heatmap row order, every instruction once")
    g.add_argument("--line", default="3,4,5", help="a,b,c: is b between a and c? empty to skip")
    g.add_argument("--n_perm", type=int, default=10000)
    g.add_argument("--seed", type=int, default=0)
    for sp in (c, s, g):
        sp.add_argument("--label", default="", help="extra title line, e.g. the model")
    args = p.parse_args()
    {"curves": curves, "scatter": scatter, "groups": groups}[args.cmd](args)


if __name__ == "__main__":
    main()
