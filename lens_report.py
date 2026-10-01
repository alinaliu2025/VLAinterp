"""
lens_report.py

Detailed logit-lens statistics for one --mode lens run, beyond the three
plots that `experiments.py --mode lens --analyze` makes. CPU only, numpy only.

    python lens_report.py runs/run2            # prints a report
    python lens_report.py runs/run2 --csv      # also writes plots/lens_report_*.csv

Reads <run>/lens/task*_init*.npz, each holding, for every env step T:
    argmax   [T, 7, 33]  bucket (0..255) each layer's lens would pick
    kl       [T, 7, 33]  KL(P_final || P_layer) in nats, over the 256 action buckets
    entropy  [T, 7, 33]  entropy of P_layer in nats (max = ln 256 = 5.55)
    buckets  [T, 7]      bucket the model actually output
Layer 0 is the embedding layer, 1..32 are decoder-layer outputs, 32 = final.

Terms used below:
    agree          layer's pick == final pick (exact bucket)
    within k       |layer's pick - final pick| <= k buckets
    first-hit      first layer whose pick equals the final pick
    settle layer   first layer from which the pick equals the final pick at
                   EVERY later layer (the point where the answer stops changing)
"""

import argparse
import csv
import glob
import os

import numpy as np

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
# norm_stats["libero_spatial"]["action"] from the finetuned checkpoint's config.
Q01 = np.array([-0.7454732, -0.6616071, -0.9375, -0.1071429, -0.2067857, -0.1842857, 0.0])
Q99 = np.array([0.9375, 0.8758929, 0.9321429, 0.1039286, 0.1767857, 0.1457143, 1.0])
N_BUCKETS = 256
LN256 = np.log(256)


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "lens", "*.npz")))
    if not files:
        raise SystemExit(f"no lens files in {run_dir}/lens")
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({k: d[k] for k in ("argmax", "kl", "entropy", "buckets")}
                   | {"success": int(d["success"]), "task": int(d["task_id"]),
                      "init": int(d["init_idx"])})
    return eps


def settle_and_first_hit(argmax):
    """argmax [N, 7, L+1] -> settle [N, 7], first_hit [N, 7] (layer indices)."""
    match = argmax == argmax[..., -1:]
    n_lay = argmax.shape[-1]
    # settle: last layer that does NOT match, plus one
    rev_nonmatch = ~match[..., ::-1]
    any_nonmatch = rev_nonmatch.any(-1)
    last_nonmatch = n_lay - 1 - rev_nonmatch.argmax(-1)
    settle = np.where(any_nonmatch, last_nonmatch + 1, 0)
    first_hit = match.argmax(-1)
    return settle, first_hit


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    eps = load(args.run_dir)
    A = np.concatenate([e["argmax"] for e in eps]).astype(np.int16)   # [N, 7, 33]
    KL = np.concatenate([e["kl"] for e in eps])
    H = np.concatenate([e["entropy"] for e in eps])
    final = A[..., -1]
    N, D, L = A.shape
    succ = np.concatenate([np.full(len(e["argmax"]), e["success"]) for e in eps]).astype(bool)
    tnorm = np.concatenate([np.linspace(0, 1, len(e["argmax"]), endpoint=False) for e in eps])
    ep_id = np.concatenate([np.full(len(e["argmax"]), i) for i, e in enumerate(eps)])
    dist = np.abs(A - final[..., None])
    settle, first_hit = settle_and_first_hit(A)
    bucket_units = (Q99 - Q01) / (N_BUCKETS - 1)   # action units per bucket

    print(f"{len(eps)} episodes, {N} env steps, {L} layer readouts "
          f"({sum(e['success'] for e in eps)} successful episodes)\n")

    # 1. what the model outputs at all (base rates)
    print("== 1. Final outputs: how varied is each action dimension?")
    print(f"{'dim':8s} {'distinct':>8s} {'top bucket':>10s} {'top share':>9s}  bucket width (action units)")
    for k in range(D):
        vals, cnt = np.unique(final[:, k], return_counts=True)
        print(f"{ACTION_NAMES[k]:8s} {len(vals):8d} {vals[cnt.argmax()]:10d} "
              f"{pct(cnt.max() / N):>9s}  {bucket_units[k]:.4f}  "
              f"(range {Q01[k]:+.3f}..{Q99[k]:+.3f})")

    # 2. agreement / near-agreement by layer
    show = [0, 8, 12, 16, 20, 24, 26, 28, 30, 31, 32]
    print("\n== 2. Exact agreement with the final bucket, by layer")
    print("layer   " + " ".join(f"{n:>8s}" for n in ACTION_NAMES))
    for l in show:
        print(f"{l:5d}   " + " ".join(f"{pct((A[:, k, l] == final[:, k]).mean()):>8s}" for k in range(D)))
    for tol in (5, 25):
        print(f"\n== 2b. Within +/-{tol} buckets of the final bucket "
              f"(= +/-{100 * tol / 255:.0f}% of each dim's range)")
        print("layer   " + " ".join(f"{n:>8s}" for n in ACTION_NAMES))
        for l in show:
            print(f"{l:5d}   " + " ".join(f"{pct((dist[:, k, l] <= tol).mean()):>8s}" for k in range(D)))

    # 3. KL and entropy
    print("\n== 3. Mean KL(P_final || P_layer) in nats  |  mean entropy of P_layer in nats "
          f"(uniform over 256 = {LN256:.2f})")
    print("layer   " + " ".join(f"{n:>7s}" for n in ACTION_NAMES) + "   |  " +
          " ".join(f"{n:>7s}" for n in ACTION_NAMES))
    for l in show:
        print(f"{l:5d}   " + " ".join(f"{KL[:, k, l].mean():7.2f}" for k in range(D)) + "   |  " +
              " ".join(f"{H[:, k, l].mean():7.2f}" for k in range(D)))

    # 4. what do early layers predict?
    print("\n== 4. Most common pick at early/mid layers (share of steps), vs. final")
    for l in (0, 8, 16):
        row = []
        for k in range(D):
            vals, cnt = np.unique(A[:, k, l], return_counts=True)
            row.append(f"{ACTION_NAMES[k]}:{vals[cnt.argmax()]}({100 * cnt.max() / N:.0f}%)")
        print(f"layer {l:2d}: " + "  ".join(row))

    # 5. settle layer
    print("\n== 5. Settle layer (from here on, every layer agrees with the final pick)")
    print(f"{'dim':8s} {'median':>6s} {'p25':>5s} {'p75':>5s} {'<=24':>7s} {'<=28':>7s} {'=32 only':>9s}"
          f"   first-hit median")
    for k in range(D):
        s = settle[:, k]
        print(f"{ACTION_NAMES[k]:8s} {np.median(s):6.0f} {np.percentile(s, 25):5.0f} "
              f"{np.percentile(s, 75):5.0f} {pct((s <= 24).mean()):>7s} {pct((s <= 28).mean()):>7s} "
              f"{pct((s == 32).mean()):>9s}   {np.median(first_hit[:, k]):.0f}")

    # 6. gripper: steady vs switching steps
    g = final[:, 6]
    g_open = g >= 128
    prev = np.roll(g_open, 1)
    first_of_ep = np.r_[True, ep_id[1:] != ep_id[:-1]]
    switch = (g_open != prev) & ~first_of_ep
    print(f"\n== 6. Gripper: steps where the open/close decision CHANGES vs. steady steps")
    print(f"open steps {pct(g_open.mean())}, switch steps: {switch.sum()} "
          f"(open->close {int((switch & ~g_open).sum())}, close->open {int((switch & g_open).sum())})")
    for name, m in (("steady", ~switch & ~first_of_ep), ("switch", switch)):
        print(f"  {name:6s} n={m.sum():5d}  settle median {np.median(settle[m, 6]):.0f}  "
              f"agree@16 {pct((A[m, 6, 16] == g[m]).mean())}  agree@20 {pct((A[m, 6, 20] == g[m]).mean())}  "
              f"agree@24 {pct((A[m, 6, 24] == g[m]).mean())}  agree@28 {pct((A[m, 6, 28] == g[m]).mean())}")

    # 7. by phase / time / outcome, for the movement dims
    move = slice(0, 6)
    def mv(m):
        return (f"settle median {np.median(settle[m, move]):4.0f}  "
                f"agree@28 {pct((A[m, move, 28] == final[m, move]).mean())}  "
                f"within5@24 {pct((dist[m, move, 24] <= 5).mean())}")
    print("\n== 7. Movement dims (x..yaw pooled), split by situation")
    print(f"  gripper open (approach)    n={g_open.sum():5d}  {mv(g_open)}")
    print(f"  gripper closed (carrying)  n={(~g_open).sum():5d}  {mv(~g_open)}")
    for lo in (0, 0.25, 0.5, 0.75):
        m = (tnorm >= lo) & (tnorm < lo + 0.25)
        print(f"  episode time {int(lo * 100):3d}-{int(lo * 100) + 25:3d}%     n={m.sum():5d}  {mv(m)}")
    print(f"  successful episodes        n={succ.sum():5d}  {mv(succ)}")
    print(f"  failed episodes            n={(~succ).sum():5d}  {mv(~succ)}")

    if args.csv:
        out = os.path.join(args.run_dir, "plots")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "lens_report_by_layer.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "dim", "agree", "within5", "within25", "mean_abs_bucket_dist",
                        "mean_kl_nats", "mean_entropy_nats"])
            for l in range(L):
                for k in range(D):
                    w.writerow([l, ACTION_NAMES[k], f"{(A[:, k, l] == final[:, k]).mean():.4f}",
                                f"{(dist[:, k, l] <= 5).mean():.4f}", f"{(dist[:, k, l] <= 25).mean():.4f}",
                                f"{dist[:, k, l].mean():.3f}", f"{KL[:, k, l].mean():.4f}",
                                f"{H[:, k, l].mean():.4f}"])
        path2 = os.path.join(out, "lens_report_settle.csv")
        with open(path2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dim", "settle_median", "settle_p25", "settle_p75", "frac_settle_le24",
                        "frac_settle_le28", "frac_settle_eq32", "first_hit_median"])
            for k in range(D):
                s = settle[:, k]
                w.writerow([ACTION_NAMES[k], np.median(s), np.percentile(s, 25), np.percentile(s, 75),
                            f"{(s <= 24).mean():.4f}", f"{(s <= 28).mean():.4f}",
                            f"{(s == 32).mean():.4f}", np.median(first_hit[:, k])])
        print(f"\n[csv] {path}\n[csv] {path2}")


if __name__ == "__main__":
    main()
