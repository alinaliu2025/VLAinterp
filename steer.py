"""
steer.py

Instruction steering at ONE position (run6). Does the task code that run5
found at the last prompt position (TDNV minimum at layer 7) actually drive the
action?

For layer l, instruction A (the prompt) and target instruction B:
    delta = mu_B[l] - mu_A[l]          (run5 cluster means, fit images only)
    add delta to the residual stream at the LAST prompt position, at the output
    of decoder layer l, in the prefill pass only; then generate the 7 action
    tokens as predict_action does.
The instruction tokens themselves are untouched, so later layers and the
action tokens can still read A from them. This is the "last position only"
variant on purpose; --positions all is left for a follow-up.

Held-out design: means come from the run5 images at --fit_inits (default 1-4);
steering is tested on the images at --test_inits (default 0).
Control: a random direction with the same norm as delta.

    python steer.py --grid runs/run5/grid/suite.npz --out_dir runs/run6        # GPU
    python steer.py --analyze --out_dir runs/run6                               # CPU

Output: <out_dir>/steer.npz, <out_dir>/plots/steer_*.{png,csv}
"""

import argparse
import csv
import os
import time

import numpy as np

from experiments import (ACTION_NAMES, DIM_COLORS, GRID, INK, MUTED, SEED,
                         SanityCheckError, generate_with_hidden, new_axes,
                         parse_range, save_fig, save_npz_atomic,
                         set_seed_everywhere, write_layer_csv)


# ---------------------------------------------------------------------------
# Step 1 (GPU): steer
# ---------------------------------------------------------------------------

class LastPositionSteer:
    """Forward hook on one decoder layer: add `delta` at the last position, prefill only."""

    def __init__(self, layer_module):
        self.delta = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.delta is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] == 1:                 # KV-cache step for an action token: leave it
            return output
        hs = hs.clone()
        hs[:, -1, :] += self.delta.to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def remove(self):
        self.handle.remove()


def steer(args):
    import torch
    from PIL import Image
    from singleVLA import BASE_REPO, build_prompt, find_decoder_layers, load_model

    set_seed_everywhere(SEED)
    rng = np.random.default_rng(SEED)
    G = np.load(args.grid)
    hidden_all = G["hidden"]                          # [K, M, L+1, d] float32 (memmap-free load)
    instructions = [str(s) for s in G["instructions"]]
    src, images = G["image_src"], G["images"]         # images: exactly what the model saw
    K, M, n_read, d = hidden_all.shape
    fit = np.where(np.isin(src[:, 1], args.fit_init_list))[0]
    test = np.where(np.isin(src[:, 1], args.test_init_list))[0]
    print(f"[info] {K} instructions; means from {len(fit)} images (inits {args.fit_inits}), "
          f"steering tested on {len(test)} images (inits {args.test_inits})")
    mu = hidden_all[:, fit].mean(axis=1)              # [K, L+1, d] float32
    run5_actions = G["actions"][:, test]              # [K, T, 7] for the reproducibility check
    del hidden_all

    device = "cuda"
    processor, model, dtype = load_model(args.model_id, BASE_REPO, device)
    unnorm_key = args.suite
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    layers_mod = find_decoder_layers(model)
    n_layers = len(layers_mod)
    if n_read != n_layers + 1:
        raise SanityCheckError(f"grid has {n_read} readouts, model {n_layers + 1}")
    vocab = int(model.vocab_size)

    # Inputs for every (instruction, test image), built once. The saved images are
    # already center-cropped (model_input_image), so they go in as they are.
    inputs = {}
    for a in range(K):
        prompt = build_prompt(instructions[a])
        for j, n in enumerate(test):
            inputs[a, j] = processor(prompt, Image.fromarray(images[n])).to(device, dtype=dtype)

    mu_t = torch.from_numpy(mu).to(device)                       # float32 on GPU
    diff_r = mu_t[None] - mu_t[:, None]                          # [A, B, L+1, d]: mu_B - mu_A
    diff_n2 = (diff_r ** 2).sum(-1)                              # [A, B, L+1]

    def run(a, j):
        with torch.no_grad():
            _, tok, h, _ = generate_with_hidden(model, inputs[a, j], unnorm_key)
        return (vocab - 1 - tok).astype(np.int16), h[0].float()   # buckets [7], h [L+1, d]

    def rep_shift(h, a, b):
        """Per readout: position along mu_A -> mu_B (0 = A's mean, 1 = B's mean)."""
        num = ((h - mu_t[a]) * diff_r[a, b]).sum(-1)
        return (num / diff_n2[a, b].clamp_min(1e-12)).cpu().numpy()

    T = len(test)
    t0 = time.time()
    base = np.zeros((K, T, 7), dtype=np.int16)
    for a in range(K):
        for j in range(T):
            base[a, j], _ = run(a, j)
    print(f"[base] {K * T} unsteered calls ({time.time() - t0:.0f}s)", flush=True)
    # Same images and prompts as run5: the buckets must reproduce run5's actions.
    ok = np.isclose(run5_actions, _actions_from_buckets(model, base, unnorm_key), rtol=0, atol=1e-6)
    if not ok.all():
        raise SanityCheckError(f"unsteered actions differ from run5 on "
                               f"{int((~ok).any(-1).sum())} of {K * T} pairs")
    print("[check] unsteered actions reproduce run5")

    L = len(args.layer_list)
    steered = np.zeros((L, K, K, T, 7), dtype=np.int16)          # [layer, A, B, image, dim]
    shift = np.full((L, K, K, T, n_read), np.nan, dtype=np.float32)
    ctrl_b = rng.integers(0, K - 1, size=(L, K, T))              # random target for the control
    ctrl_b = ctrl_b + (ctrl_b >= np.arange(K)[None, :, None])     # never B = A
    control = np.zeros((L, K, T, 7), dtype=np.int16)
    ctrl_shift = np.full((L, K, T, n_read), np.nan, dtype=np.float32)

    for li, l in enumerate(args.layer_list):
        hook = LastPositionSteer(layers_mod[l - 1])               # output of layer l = readout l
        try:
            if li == 0:
                # zero delta must change nothing; a real delta must show up at readout l
                hook.delta = torch.zeros(d, device=device)
                b0, _ = run(0, 0)
                if not np.array_equal(b0, base[0, 0]):
                    raise SanityCheckError("zero-delta hook changed the action")
                hook.delta = None
                _, h_plain = run(0, 0)
                hook.delta = (mu_t[1, l] - mu_t[0, l]).to(dtype)
                _, h_st = run(0, 0)
                err = (h_st[l] - h_plain[l] - hook.delta.float()).abs().max().item()
                if err > 0.05 * hook.delta.float().abs().max().item():
                    raise SanityCheckError(f"readout {l} moved by the wrong amount (max err {err:.3g})")
                if not torch.allclose(h_st[l - 1], h_plain[l - 1], atol=1e-3):
                    raise SanityCheckError(f"readout {l - 1} changed; hook is on the wrong layer")
                print(f"[check] hook at layer {l}: zero delta is a no-op, readout {l} "
                      f"moves by delta (max err {err:.2g}), readout {l - 1} untouched")
            for a in range(K):
                for b in range(K):
                    if b == a:
                        continue
                    hook.delta = diff_r[a, b, l].to(dtype)
                    for j in range(T):
                        steered[li, a, b, j], h = run(a, j)
                        shift[li, a, b, j] = rep_shift(h, a, b)
                for j in range(T):
                    b = int(ctrl_b[li, a, j])
                    v = torch.randn(d, generator=torch.Generator().manual_seed(
                        int(SEED * 100000 + li * 1000 + a * 10 + j))).to(device)
                    hook.delta = (v / v.norm() * diff_n2[a, b, l].sqrt()).to(dtype)
                    control[li, a, j], h = run(a, j)
                    ctrl_shift[li, a, j] = rep_shift(h, a, b)
        finally:
            hook.remove()
        print(f"[steer] layer {l}: {K * (K - 1) * T} steered + {K * T} control calls "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "steer.npz")
    save_npz_atomic(path, base=base, steered=steered, shift=shift, control=control,
                    ctrl_b=ctrl_b, ctrl_shift=ctrl_shift,
                    layers=np.array(args.layer_list), instructions=np.array(instructions),
                    test_src=src[test], positions=np.array("last"),
                    delta_norm=np.sqrt(diff_n2.cpu().numpy())[:, :, args.layer_list])
    print(f"[ok] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def _actions_from_buckets(model, buckets, unnorm_key):
    """Bucket indices [..., 7] -> unnormalized actions, exactly as predict_action decodes."""
    from experiments import decode_action_tokens
    vocab = int(model.vocab_size)
    flat = buckets.reshape(-1, 7)
    out = np.stack([decode_action_tokens(model, vocab - 1 - row.astype(np.int64), unnorm_key)
                    for row in flat])
    return out.reshape(buckets.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 (CPU): analyze
# ---------------------------------------------------------------------------

def transfer(s, a, b):
    """
    How far the steered action moved from A's action toward B's, in bucket units
    over the 7 dims: 0 = unchanged, 1 = exactly B's. nan where A and B agree.
    """
    s, a, b = (x.astype(np.float64) for x in (s, a, b))
    dab = b - a
    n2 = (dab ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n2 > 0, ((s - a) * dab).sum(-1) / n2, np.nan)


def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    D = np.load(os.path.join(args.out_dir, "steer.npz"))
    base, st, ctrl, ctrl_b = D["base"], D["steered"], D["control"], D["ctrl_b"]
    shift, ctrl_shift, layers = D["shift"], D["ctrl_shift"], D["layers"]
    L, K, _, T, _ = st.shape
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    off = ~np.eye(K, dtype=bool)
    bA = np.broadcast_to(base[None, :, None], st.shape)          # A's own action
    bB = np.broadcast_to(base[None, None, :], st.shape)          # B's action on the same image
    differs = bA != bB                                           # dims where A and B disagree

    rows = []
    for li, l in enumerate(layers):
        s, a, b, dif = st[li][off], bA[li][off], bB[li][off], differs[li][off]   # [K(K-1), T, 7]
        tr = transfer(s, a, b)
        cb = base[ctrl_b[li], np.arange(T)[None]]                # control's target actions [K, T, 7]
        ctr = transfer(ctrl[li], base, cb)
        row = {"layer": int(l),
               "transfer_mean": np.nanmean(tr), "transfer_median": np.nanmedian(tr),
               "control_transfer_mean": np.nanmean(ctr),
               "frac_closer_to_B": np.mean(np.abs(s - b).sum(-1) < np.abs(s - a).sum(-1)),
               "frac_action_changed": np.mean((s != a).any(-1)),
               "control_frac_action_changed": np.mean((ctrl[li] != base).any(-1)),
               "rep_shift_at_last_layer": np.nanmean(shift[li][off][..., -2]),
               "rep_shift_post_norm": np.nanmean(shift[li][off][..., -1])}
        for j, nm in enumerate(ACTION_NAMES):                    # flips to B, among dims that differ
            m = dif[..., j]
            row[f"match_B_{nm}"] = (s[..., j] == b[..., j])[m].mean() if m.any() else np.nan
            row[f"keep_A_{nm}"] = (s[..., j] == a[..., j])[m].mean() if m.any() else np.nan
        rows.append(row)

    out = os.path.join(plots, "steer_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[csv]  {out}")

    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    # 1. does the action move toward B?
    fig, ax = new_axes("Steering the last prompt position from A toward B (held-out images)",
                       "transfer: 0 = A's action, 1 = B's action")
    ax.plot(layers, col("transfer_mean"), "-o", color=DIM_COLORS[0], linewidth=2,
            markersize=5, label="mean_B - mean_A added")
    ax.plot(layers, col("control_transfer_mean"), "--o", color=MUTED, linewidth=1.5,
            markersize=4, label="random direction, same norm")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axhline(1, color=GRID, linewidth=1, linestyle=":")
    ax.set_xlabel("layer where the vector is added", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, os.path.join(plots, "steer_transfer.png"))

    # 2. which action dims flip to B's bucket?
    fig, ax = new_axes("Action dims that switch to B's bucket (dims where A and B differ)",
                       "fraction equal to B's bucket")
    for j, nm in enumerate(ACTION_NAMES):
        ax.plot(layers, col(f"match_B_{nm}"), "-o", color=DIM_COLORS[j], linewidth=1.6,
                markersize=4, label=nm)
    ax.set_xlabel("layer where the vector is added", color=MUTED)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    save_fig(fig, os.path.join(plots, "steer_match_by_dim.png"))

    # 3. does the representation stay moved, or wash out downstream?
    n_read = shift.shape[-1]
    fig, ax = new_axes("Where the steered state sits after the edit (0 = A's cluster, 1 = B's)",
                       "position along mean_A -> mean_B")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    for li, l in enumerate(layers):
        curve = np.nanmean(shift[li][off], axis=(0, 1))
        ax.plot(np.arange(n_read), curve, color=cmap(li / max(L - 1, 1)), linewidth=1.6,
                label=f"added at layer {l}")
        ax.plot(np.arange(n_read), np.nanmean(ctrl_shift[li], axis=(0, 1)),
                color=cmap(li / max(L - 1, 1)), linewidth=0.8, linestyle=":")
    ax.axhline(1, color=GRID, linewidth=1)
    ax.set_xlabel("readout (dotted = random-direction control)", color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, os.path.join(plots, "steer_rep_shift.png"))
    write_layer_csv(os.path.join(plots, "steer_rep_shift.csv"),
                    {f"added_at_L{l}": np.nanmean(shift[li][off], axis=(0, 1))
                     for li, l in enumerate(layers)})

    print(f"\n{'layer':>5s} {'transfer':>9s} {'control':>8s} {'closer B':>9s} "
          f"{'changed':>8s} {'x->B':>6s} {'rep@31':>7s}")
    for r in rows:
        print(f"{r['layer']:5d} {r['transfer_mean']:9.3f} {r['control_transfer_mean']:8.3f} "
              f"{r['frac_closer_to_B']:9.2f} {r['frac_action_changed']:8.2f} "
              f"{r['match_B_x']:6.2f} {r['rep_shift_at_last_layer']:7.2f}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid", default="runs/run5/grid/suite.npz",
                   help="run5 grid: images, instructions, hidden states for the means")
    p.add_argument("--layers", default="3,5,7,9,11,15,23,31",
                   help="decoder layers to steer at (1..31; readout l = output of layer l)")
    p.add_argument("--fit_inits", default="1-4")
    p.add_argument("--test_inits", default="0")
    p.add_argument("--model_id", default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--analyze", action="store_true", help="CPU analysis of saved data")
    args = p.parse_args()
    args.layer_list = parse_range(args.layers)
    args.fit_init_list = parse_range(args.fit_inits)
    args.test_init_list = parse_range(args.test_inits)
    if set(args.fit_init_list) & set(args.test_init_list):
        raise SystemExit("[error] fit and test inits overlap")
    if any(not 1 <= l <= 31 for l in args.layer_list):
        raise SystemExit("[error] steer layers must be 1..31 (32 is the post-norm readout)")
    if args.analyze:
        analyze(args)
    else:
        steer(args)


if __name__ == "__main__":
    main()
