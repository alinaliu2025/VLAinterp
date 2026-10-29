"""
tdnv_iv_extract.py

TDNV image vs. instruction, step 1 (GPU): collect hidden states.

Images: the first frame (after the objects settle) of every libero_spatial
layout task in the config, at every init state in the config. The 10 layouts
share the same objects (two black bowls, plate, ramekin, cookie box, wooden
cabinet, stove) but place the bowls differently.

For every instruction set and every (instruction, image) pair, one forward
pass (predict_action's exact path, via generate_with_hidden) and the hidden
state at the last prompt position, the one that produces the first action
token, at all L+1 readouts (embeddings + 32 layers). Stored as float32.

    python tdnv_iv_extract.py --config runs/run5/config.json --out_dir runs/run5

Output: <out_dir>/grid/<set>.npz with
    hidden        [K_instr, M_images, L+1, d] float32
    actions       [K_instr, M_images, 7]
    instructions  [K_instr]
    images        [M_images, 224, 224, 3]   exactly what the model saw
    image_src     [M_images, 2]             (layout task, init index)
Then run tdnv_iv_analyze.py (CPU).
"""

import argparse
import json
import os
import time

import numpy as np

from experiments import (SEED, SanityCheckError, generate_with_hidden, get_tf,
                         model_input_image, run_episode, save_npz_atomic,
                         set_seed_everywhere, verify_against_predict_action)


def first_frames(suite, layout_tasks, inits):
    """First frame of each (layout task, init) after settling; no model needed."""
    images, src = [], []
    for t in layout_tasks:
        task = suite.get_task(t)
        init_states = suite.get_task_init_states(t)
        for i in inits:
            # One policy step with a no-op; the frame is recorded before it.
            _, _, frames = run_episode(task, init_states[i], lambda img, step: [0] * 6 + [1], 1)
            images.append(frames[0])
            src.append((t, i))
        print(f"[img] layout task {t}: {len(inits)} frames  ({task.language})", flush=True)
    return images, src


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sets", nargs="*", default=None, help="instruction sets (default: all)")
    args = p.parse_args()
    cfg = json.load(open(args.config))

    import torch
    from libero.libero import benchmark
    from singleVLA import BASE_REPO, build_prompt, find_decoder_layers, load_model

    get_tf()
    set_seed_everywhere(cfg.get("seed", SEED))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, dtype = load_model(cfg["model_id"], BASE_REPO, device)
    # "suite" picks the scenes; "unnorm_key" (default: the suite) picks the action
    # stats, so another checkpoint can see the same scenes. Hidden states don't use it.
    unnorm_key = cfg.get("unnorm_key", cfg["suite"])
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in model.norm_stats:
        raise SanityCheckError(f"unnorm_key {unnorm_key} not in {sorted(model.norm_stats)}")
    print(f"[info] unnorm_key = {unnorm_key}", flush=True)
    n_layers = len(find_decoder_layers(model))

    suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
    t0 = time.time()
    images, src = first_frames(suite, cfg["layout_tasks"], cfg["inits"])
    print(f"[info] {len(images)} images ({time.time() - t0:.0f}s)", flush=True)
    pil = [model_input_image(im) for im in images]
    model_imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in pil])

    os.makedirs(os.path.join(args.out_dir, "grid"), exist_ok=True)
    verified = False
    for name in args.sets or list(cfg["instruction_sets"]):
        instructions = cfg["instruction_sets"][name]
        K, M = len(instructions), len(pil)
        hidden = None
        actions = np.zeros((K, M, 7), dtype=np.float32)
        t0 = time.time()
        for k, ins in enumerate(instructions):
            prompt = build_prompt(ins)
            for n, im in enumerate(pil):
                inputs = processor(prompt, im).to(device, dtype=dtype)
                with torch.no_grad():
                    action, _, h, _ = generate_with_hidden(model, inputs, unnorm_key)
                if not verified:
                    verify_against_predict_action(model, inputs, unnorm_key, action)
                    verified = True
                if h.shape[1] != n_layers + 1:
                    raise SanityCheckError(f"expected {n_layers + 1} readouts, got {h.shape[1]}")
                if hidden is None:
                    hidden = np.zeros((K, M, n_layers + 1, h.shape[-1]), dtype=np.float32)
                hidden[k, n] = h[0].float().cpu().numpy()     # last prompt position
                actions[k, n] = np.asarray(action, dtype=np.float32)
            print(f"[{name}] {k + 1}/{K} instructions x {M} images "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if not np.isfinite(hidden).all():
            raise SanityCheckError(f"non-finite hidden states in set {name}")
        path = os.path.join(args.out_dir, "grid", f"{name}.npz")
        save_npz_atomic(path, hidden=hidden, actions=actions,
                        instructions=np.array(instructions), images=model_imgs,
                        image_src=np.array(src, dtype=np.int32),
                        model_id=np.array(cfg["model_id"]), dtype=np.array(str(dtype)))
        print(f"[ok] saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
