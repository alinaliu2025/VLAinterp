"""
instruction_swap.py

Dr. Zhu's experiment: hold the IMAGES fixed, swap the LANGUAGE instruction, and
compare OpenVLA's actions and hidden states.

    For each instruction L_k (k = 1..K) and each image I_n (n = 1..N):
        h_{k,n}  = hidden state at the position that produces the first action
                   token, at every layer (the circled h on the whiteboard)
        a_{k,n}  = the 7-DoF action the model outputs

    Per layer, per instruction:
        mean_k = (1/N) sum_n h_{k,n}
        Var_k  = (1/N) sum_n ||h_{k,n} - mean_k||^2

    Per layer, per pair of instructions:
        R(k1, k2) = (Var_k1 + Var_k2) / ||mean_k1 - mean_k2||^2

    This is the lab's TDNV (Jiang, Dong, Zhou & Zhu 2025, "From Compression
    to Expression", eq. 4) with tasks = instructions and samples = images:
    TDNV = average over pairs of R / 2. By default features are first centered
    on the global mean and scaled to unit norm, as in the lab's code (--raw to skip).

    R << 1  : the instruction moves h more than the image does (language matters)
    R >~ 1  : the two clouds overlap; the image dominates, the wording is ignored

Builds on experiments.py (same official preprocessing, same verified
generate_with_hidden path, same sanity checks). Two steps:

  1) GPU: collect hidden states and actions
       python instruction_swap.py --task 0 --inits 0-49 --out_dir runs/run4
     Images: the first frame of each LIBERO init state of --task (after the
     objects settle). Add --frame_stride S --max_steps M to also take every S-th
     frame of a rollout driven by the task's own instruction (more arm poses).

     Instructions (default): all 10 libero_spatial instructions, applied to the
     SAME images. Add your own with --extra "text one" --extra "text two"
     (paraphrases are the control: their R should be LARGE).

  2) CPU: analysis, no model needed
       python instruction_swap.py --analyze --out_dir runs/run4 --task 0

Output: <out_dir>/swap/task{t}.npz and <out_dir>/plots/swap_*.{png,csv}
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from experiments import (ACTION_NAMES, MAX_STEPS, SEED, SanityCheckError,
                         action_token_ids, generate_with_hidden, get_tf,
                         model_input_image, new_axes, parse_range, run_episode,
                         save_fig, save_npz_atomic, set_seed_everywhere,
                         verify_against_predict_action, write_layer_csv)


# ---------------------------------------------------------------------------
# Step 1 (GPU): collect
# ---------------------------------------------------------------------------

def collect(args):
    import torch
    from libero.libero import benchmark
    from singleVLA import (BASE_REPO, build_prompt, find_decoder_layers,
                           get_final_norm_and_head, load_model)

    get_tf()
    set_seed_everywhere(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id or f"openvla/openvla-7b-finetuned-{args.suite.replace('_', '-')}"
    processor, model, dtype = load_model(model_id, BASE_REPO, device)

    unnorm_key = args.suite
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"{unnorm_key} not in {list(model.norm_stats)}"

    n_layers = len(find_decoder_layers(model))
    norm, head = get_final_norm_and_head(model)
    ids = action_token_ids(model, processor)                       # bucket b -> token id
    W_action = head.weight[torch.as_tensor(ids, device=head.weight.device)].float().cpu().numpy()
    norm_w = norm.weight.float().cpu().numpy() if norm is not None else None

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    init_states = suite.get_task_init_states(args.task)

    # ---- instructions: suite instructions (own task first) + user extras ----
    instructions, labels = [task.language], [f"own(t{args.task})"]
    if not args.no_suite_instructions:
        for t in range(suite.n_tasks):
            if t != args.task:
                instructions.append(suite.get_task(t).language)
                labels.append(f"t{t}")
    for i, e in enumerate(args.extra or []):
        instructions.append(e)
        labels.append(f"extra{i}")
    print("[info] instructions:")
    for lab, ins in zip(labels, instructions):
        print(f"   {lab:9s} {ins}")

    # ---- images: frames from rollouts under the task's own instruction ----
    own_prompt = build_prompt(task.language)

    def own_policy(img_224, step):
        inputs = processor(own_prompt, model_input_image(img_224)).to(device, dtype=dtype)
        with torch.no_grad():
            return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    images, image_src = [], []          # 224x224 pre-crop frames, (init, step)
    t0 = time.time()
    for init_idx in args.init_list:
        _, _, frames = run_episode(task, init_states[init_idx], own_policy, args.max_steps)
        for s in range(0, len(frames), args.frame_stride):
            images.append(frames[s])
            image_src.append((init_idx, s))
    print(f"[info] {len(images)} images from {len(args.init_list)} init states "
          f"({time.time() - t0:.0f}s)")

    # ---- the experiment: every instruction x every image ----
    K, N = len(instructions), len(images)
    d = W_action.shape[1]
    hidden = np.zeros((K, N, n_layers + 1, d), dtype=np.float16)
    hidden7 = np.zeros((K, N, 7, n_layers + 1, d), dtype=np.float16) if args.all_tokens else None
    buckets = np.zeros((K, N, 7), dtype=np.uint8)
    actions = np.zeros((K, N, 7), dtype=np.float32)
    model_imgs = np.stack([np.asarray(model_input_image(im), dtype=np.uint8) for im in images])

    t0 = time.time()
    for k, ins in enumerate(instructions):
        prompt = build_prompt(ins)
        for n, im in enumerate(images):
            inputs = processor(prompt, model_input_image(im)).to(device, dtype=dtype)
            with torch.no_grad():
                action, token_ids, h, _ = generate_with_hidden(model, inputs, unnorm_key)
            if k == 0 and n == 0:
                verify_against_predict_action(model, inputs, unnorm_key, action)
            if h.shape[1] != n_layers + 1:
                raise SanityCheckError(f"expected {n_layers + 1} hidden states, got {h.shape[1]}")
            hidden[k, n] = h[0].to(torch.float16).cpu().numpy()
            if hidden7 is not None:
                hidden7[k, n] = h.to(torch.float16).cpu().numpy()
            buckets[k, n] = (int(model.vocab_size) - 1 - token_ids).astype(np.uint8)
            actions[k, n] = np.asarray(action, dtype=np.float32)
        print(f"[swap] {labels[k]:9s} done: {N} images ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    os.makedirs(os.path.join(args.out_dir, "swap"), exist_ok=True)
    path = os.path.join(args.out_dir, "swap", f"task{args.task}.npz")
    extra = {"hidden7": hidden7} if hidden7 is not None else {}
    save_npz_atomic(
        path,
        hidden=hidden,                        # [K, N, L+1, d] fp16
        buckets=buckets,                      # [K, N, 7]
        actions=actions,                      # [K, N, 7] unnormalized, pre-gripper-processing
        instructions=np.array(instructions), labels=np.array(labels),
        images=model_imgs,                    # [N, 224, 224, 3] exactly what the model saw
        image_src=np.array(image_src, dtype=np.int32),   # [N, 2] (init_idx, step)
        W_action=W_action,                    # [256, d] lm_head rows, bucket order
        norm_w=norm_w if norm_w is not None else np.zeros(0),
        task_id=np.int32(args.task), **extra)
    print(f"[ok] saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Step 2 (CPU): analyze
# ---------------------------------------------------------------------------

def separability(H):
    """
    H: [K, N, d] float64 at one layer.
    Returns var [K], means [K, d], R [K, K] with R[k,k] = nan.
    """
    means = H.mean(axis=1)
    var = ((H - means[:, None]) ** 2).sum(-1).mean(-1)
    dist2 = ((means[:, None] - means[None]) ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (var[:, None] + var[None]) / dist2
    np.fill_diagonal(R, np.nan)
    return var, means, R


def remove_mean_normalize(H):
    """
    The lab's preprocessing for TDNV (Jiang, Dong, Zhou & Zhu 2025, eq. 4 code):
    subtract the global mean over ALL samples (every instruction x image), then
    scale each sample to unit L2 norm. H: [K, N, d] -> same shape.
    """
    K, N, d = H.shape
    X = H.reshape(K * N, d)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X.reshape(K, N, d)


def tdnv(var, means):
    """
    Eq. 4 of "From Compression to Expression" (2505.17322), averaged over the
    K(K-1) ordered pairs as in the lab's compute_TDNV:
        TDNV = mean_{t != t'} (var_t + var_t') / (2 ||mean_t - mean_t'||^2)
    Note the factor 2: TDNV for one pair = R / 2.
    """
    K = len(var)
    dist2 = ((means[:, None] - means[None]) ** 2).sum(-1)
    off = ~np.eye(K, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(((var[:, None] + var[None]) / (2 * dist2))[off].mean())


def action_subspace_fraction(diff, W_eff_basis):
    """Fraction of ||diff||^2 inside the span of the action readout directions."""
    proj = W_eff_basis @ diff
    return float((proj ** 2).sum() / (diff ** 2).sum())


def analyze(args):
    path = os.path.join(args.out_dir, "swap", f"task{args.task}.npz")
    if not os.path.isfile(path):
        sys.exit(f"[error] {path} not found; run the collect step first")
    D = np.load(path)
    hidden, labels = D["hidden"], list(D["labels"])
    K, N, n_lay, d = hidden.shape
    print(f"[info] {K} instructions x {N} images, {n_lay} layer readouts, d={d}")

    # Orthonormal basis of the action readout space. The final RMSNorm scales
    # each coordinate by norm_w before lm_head, so the effective readout rows
    # are W_action * norm_w.
    W = D["W_action"].astype(np.float64)
    if D["norm_w"].size:
        W = W * D["norm_w"].astype(np.float64)[None]
    Q = np.linalg.qr(W.T)[0].T                   # [256, d], orthonormal rows
    chance = Q.shape[0] / d

    R_all = np.zeros((n_lay, K, K))
    var_all = np.zeros((n_lay, K))
    tdnv_all = np.zeros(n_lay)                   # one number per layer, all instructions
    frac_all = np.zeros((n_lay, K))              # own vs each other instruction
    for l in range(n_lay):
        H_raw = hidden[:, :, l].astype(np.float64)
        H = H_raw if args.raw else remove_mean_normalize(H_raw)
        var, means, R = separability(H)
        R_all[l], var_all[l] = R, var
        tdnv_all[l] = tdnv(var, means)
        # The readout-geometry question uses the RAW residual stream, because
        # that is what the final norm and lm_head actually act on.
        raw_means = H_raw.mean(axis=1)
        for k in range(1, K):
            frac_all[l, k] = action_subspace_fraction(raw_means[0] - raw_means[k], Q)

    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    layers = np.arange(n_lay)
    tag = f"task{args.task}" + ("_raw" if args.raw else "")
    feat = "raw features" if args.raw else "centered + unit-norm features"

    # 0. TDNV over all instructions (the lab's metric, one curve)
    fig, ax = new_axes(f"TDNV across {K} instructions ({tag}, N={N} images, {feat})",
                       "TDNV (log scale)")
    ax.plot(layers[1:], tdnv_all[1:], color="#2a78d6", linewidth=2.2)
    ax.set_yscale("log")
    save_fig(fig, os.path.join(plots, f"swap_TDNV_{tag}.png"))
    write_layer_csv(os.path.join(plots, f"swap_TDNV_{tag}.csv"), {"TDNV": tdnv_all})

    # 1. R vs layer, own instruction vs every other instruction
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps["viridis"]
    fig, ax = new_axes(f"Instruction swap ({tag}, N={N} images): "
                       "R = (Var1+Var2) / ||mean1-mean2||²", "R (log scale)")
    for k in range(1, K):
        style = "--" if labels[k].startswith("extra") else "-"
        ax.plot(layers, R_all[:, 0, k], style, color=cmap(k / max(K - 1, 1)),
                linewidth=1.5, label=f"own vs {labels[k]}")
    ax.axhline(1, color="#898781", linestyle=":", linewidth=1.2, label="R = 1")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, os.path.join(plots, f"swap_R_by_layer_{tag}.png"))
    write_layer_csv(os.path.join(plots, f"swap_R_by_layer_{tag}.csv"),
                    {f"own_vs_{labels[k]}": R_all[:, 0, k] for k in range(1, K)})

    # 2. fraction of the mean shift that lies in the action readout space
    fig, ax = new_axes(f"Where does the instruction shift point? ({tag})",
                       "fraction of ||mean_own - mean_k||² in action-token span")
    for k in range(1, K):
        style = "--" if labels[k].startswith("extra") else "-"
        ax.plot(layers, frac_all[:, k], style, color=cmap(k / max(K - 1, 1)),
                linewidth=1.5, label=f"own vs {labels[k]}")
    ax.axhline(chance, color="#898781", linestyle=":", linewidth=1.2,
               label=f"random direction ({chance:.3f})")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, os.path.join(plots, f"swap_action_subspace_{tag}.png"))
    write_layer_csv(os.path.join(plots, f"swap_action_subspace_{tag}.csv"),
                    {f"own_vs_{labels[k]}": frac_all[:, k] for k in range(1, K)})

    # 3. full K x K R matrix at one layer
    lay = args.layer if args.layer is not None else n_lay - 2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.log10(R_all[lay]), cmap="viridis")
    ax.set_xticks(range(K), labels, rotation=60, fontsize=7)
    ax.set_yticks(range(K), labels, fontsize=7)
    ax.set_title(f"log10 R between instructions, layer {lay} ({tag})", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, label="log10 R  (lower = more separated)")
    save_fig(fig, os.path.join(plots, f"swap_R_matrix_L{lay}_{tag}.png"))

    # 4. behavior: does the ACTION change when only the words change?
    b, a = D["buckets"].astype(int), D["actions"]
    rows = []
    for k in range(1, K):
        changed = (b[k] != b[0])                          # [N, 7]
        row = {"pair": f"own_vs_{labels[k]}",
               "instruction": str(D["instructions"][k]),
               "frac_images_any_dim_changed": changed.any(-1).mean()}
        for j, nm in enumerate(ACTION_NAMES):
            row[f"frac_changed_{nm}"] = changed[:, j].mean()
            row[f"mean_abs_bucket_diff_{nm}"] = np.abs(b[k, :, j] - b[0, :, j]).mean()
        row[f"R_layer{lay}"] = R_all[lay, 0, k]
        row["R_min_over_layers"] = np.nanmin(R_all[:, 0, k])
        row["layer_of_min_R"] = int(np.nanargmin(R_all[:, 0, k]))
        rows.append(row)
    out = os.path.join(plots, f"swap_summary_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[csv]  {out}")

    best = int(np.nanargmin(tdnv_all[1:])) + 1
    print(f"\n[TDNV] minimum {tdnv_all[best]:.4f} at layer {best} ({feat})")
    print(f"[summary] own instruction: {D['instructions'][0]}")
    print(f"{'pair':16s} {'R@L' + str(lay):>9s} {'min R':>8s} {'@layer':>6s} {'action changed':>15s}")
    for r in rows:
        print(f"{r['pair']:16s} {r[f'R_layer{lay}']:9.3f} {r['R_min_over_layers']:8.3f} "
              f"{r['layer_of_min_R']:6d} {r['frac_images_any_dim_changed']:14.0%}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task", type=int, default=0, help="whose scenes supply the images")
    p.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    p.add_argument("--inits", default="0-49", help="LIBERO init states to use as images")
    p.add_argument("--max_steps", type=int, default=1,
                   help="rollout length per init under the own instruction (1 = first frame only)")
    p.add_argument("--frame_stride", type=int, default=1000,
                   help="keep every Nth rollout frame (default keeps only frame 0)")
    p.add_argument("--extra", action="append",
                   help="extra instruction (repeatable), e.g. a paraphrase control")
    p.add_argument("--no_suite_instructions", action="store_true",
                   help="only the own instruction + --extra ones")
    p.add_argument("--all_tokens", action="store_true",
                   help="also save hidden states at all 7 action-token positions (7x disk)")
    p.add_argument("--model_id", default=None)
    p.add_argument("--analyze", action="store_true", help="CPU analysis of saved data")
    p.add_argument("--raw", action="store_true",
                   help="skip the global-mean removal + unit-norm step (default: apply it, "
                        "as in the lab's TDNV code)")
    p.add_argument("--layer", type=int, default=None,
                   help="layer for the R matrix and summary (default: second to last)")
    args = p.parse_args()
    args.init_list = parse_range(args.inits)
    if args.analyze:
        analyze(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
