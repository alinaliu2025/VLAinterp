"""
peek.py -- look at a capture from the cluster, on your laptop.

The cluster writes .pt, which needs torch to open. Convert it to .npz on OSC
first (see docs) so that all you need locally is numpy.

    python3 peek.py data/static_probe.npz
    python3 peek.py data/static_probe.npz --plot
"""

import argparse

import numpy as np

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def name_of(k):
    return ACTION_NAMES[k] if k < len(ACTION_NAMES) else f"a{k}"


def cosine_to_last(step_matrix):
    """step_matrix: [layers, d_model]. Cosine of each layer with the last one."""
    ref = step_matrix[-1]
    denom = np.linalg.norm(step_matrix, axis=-1) * np.linalg.norm(ref) + 1e-8
    return step_matrix @ ref / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to a .npz written from a cluster capture")
    ap.add_argument("--plot", action="store_true", help="also save a heatmap png")
    args = ap.parse_args()

    d = np.load(args.path)
    action = d["action"]
    hidden = d["hidden"].astype(np.float32)   # [gen_steps, layers, d_model]
    layers = list(d["layer_indices"])

    n_steps, n_layers, d_model = hidden.shape
    print(f"hidden  {hidden.shape}   (gen_steps, layers, d_model)")
    print(f"layers  {layers}")
    print()

    # Generation step k is the pass that produced action token k, so the rows
    # of `hidden` line up one-to-one with the action dimensions.
    print("predicted action")
    for i, v in enumerate(np.asarray(action).ravel()):
        print(f"  {name_of(i):>8}  {v: .5f}")
    print()

    header = "           " + " ".join(f"{l:>7d}" for l in layers)

    print("residual-stream L2 norm      (row = action dim, col = decoder layer)")
    print(header)
    norms = np.linalg.norm(hidden, axis=-1)
    for k in range(n_steps):
        print(f"  {name_of(k):>8}  " + " ".join(f"{v:7.1f}" for v in norms[k]))
    print()

    print("cosine similarity to the last hooked layer")
    print(header)
    sim = np.stack([cosine_to_last(hidden[k]) for k in range(n_steps)])
    for k in range(n_steps):
        print(f"  {name_of(k):>8}  " + " ".join(f"{v:7.3f}" for v in sim[k]))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 4))
        im = ax.imshow(sim, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_layers), [str(l) for l in layers])
        ax.set_yticks(range(n_steps), [name_of(k) for k in range(n_steps)])
        ax.set_xlabel("decoder layer")
        ax.set_ylabel("action dimension")
        ax.set_title("cosine similarity to final hooked layer")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out = args.path.rsplit(".", 1)[0] + "_cosine.png"
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
