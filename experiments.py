"""
experiments.py

Batch experiments on OpenVLA + LIBERO, built on top of singleVLA.py.

    --mode rollout          success rate only (plain predict_action rollouts)
    --mode lens             rollouts + logit-lens data + hidden states for probing
    --mode lens --analyze   plots from the saved logit-lens data (CPU, no model)
    --mode probe            ridge probes on the saved hidden states (CPU, no model)
    --mode patchlens        --mode lens, plus a logit lens on the 256 image-patch
                            positions over the text vocabulary (perception)
    --mode tunedlens        tuned lens: train a per-layer affine translator in front
                            of the frozen read-out on saved hidden states (GPU, no
                            rollouts); --src_dir points at the --mode lens run
    --mode tunedlens --analyze   replot saved translator results (CPU)

Modes say WHAT the script does. Runs are numbered jobs (run0, run1, ...), each
with its own folder runs/runN/ passed as --out_dir; see runs/RUNS.md.

Episodes are indexed by (task_id, init_idx). The model decodes greedily and the
simulator is deterministic, so the ONLY thing that differs between episodes of
the same task is which of LIBERO's 50 fixed initial states we start from.

Preprocessing and the rollout loop are copied from OpenVLA's official eval
(github.com/openvla/openvla, experiments/robot/libero/run_libero_eval.py and its
utils). That repo is not installed on OSC, so the relevant pieces are inlined
below rather than imported. The image steps need TensorFlow, exactly as the
official code does:

    pip install tensorflow-cpu==2.15.1

Deviations from the official eval, all deliberate:
  - fp16 + SDPA instead of bf16 + flash-attn (V100 has neither bf16 nor FA2).
  - A fresh env per episode instead of one per task, so a resumed SLURM job
    produces exactly what an uninterrupted one would.
  - Only env.step() is wrapped in try/except (the official code wraps the
    whole step, which would also swallow our own sanity-check failures).

Output layout (out_dir is required, e.g. runs/run1):
    episodes.csv                          one row per finished episode
    videos/task{t}_init{i}_{success|fail}.mp4
    lens/task{t}_init{i}.npz              --mode lens
    hidden/task{t}_init{i}.npz            --mode lens
    patchlens/task{t}_init{i}.npz         --mode patchlens
    patch_hidden/task{t}_init{i}.npz      --mode patchlens (every PATCH_HIDDEN_EVERY steps)
    plots/                                --analyze, --mode probe, --mode tunedlens
    tunedlens/readout.npz                 --mode tunedlens: cached lm_head action rows + final norm
    tunedlens/results.npz                 --mode tunedlens: metrics, out-of-fold argmax
"""

import argparse
import csv
import glob
import os
import sys
import time

# Must be set before anything imports mujoco/robosuite (same as singleVLA.py).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

# torch, transformers, LIBERO, TensorFlow, sklearn and matplotlib are all
# imported inside the functions that need them. That way --analyze and --mode probe
# run on a CPU node (or your laptop) with only numpy + sklearn + matplotlib.

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# From run_libero_eval.py: the longest training demo per suite, plus margin.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10               # official: let objects fall and settle first
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
ENV_RESOLUTION = 256              # official: camera renders at 256x256
MODEL_IMAGE_SIZE = 224            # official get_image_resize_size() for openvla
CENTER_CROP_SCALE = 0.9           # official crop area fraction (side = sqrt(0.9))
SEED = 7                          # official default seed
EMPTY_TOKEN_ID = 29871            # predict_action appends this after "Out:"

CSV_FIELDS = ["task_id", "init_idx", "instruction", "success", "num_steps"]

# --mode patchlens. Prismatic's forward builds the sequence as
# [BOS] + 256 projected image patches + the rest of the prompt
# (modeling_prismatic.py: torch.cat([input_embeddings[:, :1], projected_patch_embeddings,
# input_embeddings[:, 1:]])), so the patches sit at positions 1..256.
N_IMAGE_PATCHES = 256
PATCH_TOPK = 5
PATCH_HIDDEN_EVERY = 20           # also save raw patch hidden states every N steps
# Words scored at every patch and layer. Single SentencePiece tokens only
# (checked with the openvla-7b tokenizer); "bowl" itself splits into
# "▁bow" + "l", so both "▁Bowl" (one token) and "▁bow" (first piece) are kept.
# The last five are unrelated controls.
PATCH_CANDIDATES = ["▁Bowl", "▁bow", "▁plate", "▁cookie", "▁box", "▁cabinet", "▁table",
                    "▁Table", "▁robot", "▁arm", "▁black", "▁wooden", "▁kitchen", "▁floor",
                    "▁wall", "▁container", "▁pot",
                    "▁car", "▁dog", "▁tree", "▁house", "▁sky"]


class SanityCheckError(RuntimeError):
    """A check that means the saved data cannot be trusted. Never swallowed."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_range(spec):
    """'0-9' -> [0..9], '3' -> [3], '0-2,5,7-8' -> [0,1,2,5,7,8]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def episode_stem(task_id, init_idx):
    return f"task{task_id}_init{init_idx}"


def save_npz_atomic(path, **arrays):
    """
    Write to a temp file, then rename. If a SLURM job is killed mid-write, the
    half-written file never appears under its real name, so resume logic that
    checks "does the file exist" can't be fooled by a truncated file.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def set_seed_everywhere(seed):
    """Copied from OpenVLA's robot_utils.py."""
    import random
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Official image preprocessing (TensorFlow, copied from OpenVLA)
# ---------------------------------------------------------------------------

_TF = None


def get_tf():
    """Import TensorFlow once, keep it off the GPU, fail with a clear message."""
    global _TF
    if _TF is None:
        try:
            import tensorflow as tf
        except ImportError:
            sys.exit(
                "[error] TensorFlow is required to match OpenVLA's official image "
                "preprocessing. Install it into vla310 with:\n"
                "    pip install tensorflow-cpu==2.15.1"
            )
        try:
            # We only use TF for a few image ops. Don't let it grab V100 memory.
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _TF = tf
    return _TF


def resize_image_official(img, size=MODEL_IMAGE_SIZE):
    """
    libero_utils.resize_image: JPEG encode/decode (as the RLDS dataset builder
    did at training time), then lanczos3 resize with antialiasing.
    """
    tf = get_tf()
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (size, size), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def get_libero_image(obs):
    """
    libero_utils.get_libero_image: rotate 180 degrees (not just a vertical flip)
    to match training, then resize to 224. This 224 image is also what the
    official eval writes to its replay videos.
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return resize_image_official(img, MODEL_IMAGE_SIZE)


def center_crop_official(img_uint8, crop_scale=CENTER_CROP_SCALE):
    """
    openvla_utils.crop_and_resize + the center_crop branch of get_vla_action.
    The finetuned checkpoint was trained with random crops covering 90% of the
    image area, so at test time we take the central 90% and resize back to 224.
    Note the crop SIDE is sqrt(0.9), because 0.9 is an area fraction.
    """
    tf = get_tf()
    image = tf.convert_to_tensor(img_uint8)
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)       # [0, 1]

    image = tf.expand_dims(image, axis=0)                         # batch of 1
    side = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(1,))
    offset = (1 - side) / 2
    boxes = tf.stack([offset, offset, offset + side, offset + side], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(1), (224, 224))
    image = image[0]

    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return image.numpy()


def model_input_image(img_224):
    """The PIL image that actually goes into the processor."""
    from PIL import Image
    image = Image.fromarray(img_224).convert("RGB")
    image = Image.fromarray(center_crop_official(np.array(image))).convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Official action post-processing and env construction
# ---------------------------------------------------------------------------

def postprocess_action(action):
    """
    robot_utils.normalize_gripper_action(binarize=True) then
    invert_gripper_action. The model outputs gripper in [0, 1] with 1 = open;
    LIBERO wants -1 = open, +1 = close.
    """
    action = np.array(action, dtype=np.float64, copy=True)
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] = action[..., -1] * -1.0
    return action


def make_env(task):
    """libero_utils.get_libero_env. Seed 0 matters even with a fixed init state."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
    )
    env.seed(0)
    return env


def object_pos_keys(obs):
    """
    Absolute object positions in the obs dict. LIBERO also exposes keys that
    end in _pos but are not object positions: robot0_* (robot state) and
    <obj>_to_robot0_eef_pos (object relative to the gripper). Skip both.
    """
    return sorted(
        k for k in obs
        if k.endswith("_pos") and not k.startswith("robot0_") and "_to_" not in k
    )


# ---------------------------------------------------------------------------
# Model: action tokens, generate-with-hidden-states, logit lens
# ---------------------------------------------------------------------------

def action_token_ids(model, processor):
    """
    Token id for each of the 256 action buckets.

    OpenVLA's ActionTokenizer overwrites the last n_bins tokens of the Llama
    vocab: continuous value -> bin d in [1, 256] -> token id vocab_size - d.
    predict_action inverts that with bucket = vocab_size - id - 1. So bucket b
    (0..255) is token vocab_size - 1 - b, and bucket 0 is the highest id.

    model.vocab_size is the text vocab with the pad-to-multiple rows removed
    (32064 - 64 = 32000), which is what predict_action decodes against.
    """
    vocab = int(model.vocab_size)
    n_bins = int(model.config.n_action_bins)
    tok_vocab = int(processor.tokenizer.vocab_size)
    if tok_vocab != vocab:
        raise SanityCheckError(
            f"tokenizer.vocab_size={tok_vocab} but model.vocab_size={vocab}; the "
            "ActionTokenizer contract assumes they are equal.")
    ids = vocab - 1 - np.arange(n_bins)
    # ActionTokenizer.action_token_begin_idx is the exclusive lower bound.
    begin_idx = tok_vocab - (n_bins + 1)
    assert ids.min() == begin_idx + 1 and ids.max() == vocab - 1
    print(f"[info] action tokens: {n_bins} buckets, ids {ids.min()}..{ids.max()} "
          f"(ActionTokenizer.action_token_begin_idx={begin_idx}); "
          f"bucket b <-> token {vocab - 1} - b; lm_head has "
          f"{model.config.text_config.vocab_size} outputs")
    return ids


def decode_action_tokens(model, token_ids, unnorm_key):
    """The tail of predict_action: token ids -> unnormalized continuous action."""
    discretized = model.vocab_size - token_ids
    discretized = np.clip(discretized - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


def generate_with_hidden(model, inputs, unnorm_key, image_hidden=False):
    """
    predict_action, line for line, except generate() is asked to also return
    every layer's hidden states. Deliberately keeps predict_action's quirk of
    appending the empty token to input_ids but NOT extending attention_mask.

    Returns:
        action     [7]            unnormalized, before gripper post-processing
        token_ids  [7]            generated token ids
        hidden     [7, L+1, d]    hidden state at the position that PRODUCES
                                  each action token, all layers, on GPU
        img        [L+1, 256, d]  (only if image_hidden) hidden states at the
                                  256 image-patch positions, from the prefill
                                  pass; None otherwise
    """
    import torch
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN_ID]).long(), dim=0).to(input_ids.device)),
            dim=1,
        )
    rest = {k: v for k, v in inputs.items() if k != "input_ids"}
    n = model.get_action_dim(unnorm_key)

    out = model.generate(
        input_ids,
        max_new_tokens=n,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        **rest,
    )
    token_ids = out.sequences[0, -n:].cpu().numpy()
    action = decode_action_tokens(model, token_ids, unnorm_key)

    # out.hidden_states has one entry per generated token. Entry 0 is the
    # prefill pass over the whole prompt: its LAST position produced token 0.
    # Entries 1..6 are single-position passes (KV cache), each producing the
    # next token. Each entry is a tuple of L+1 tensors: the embeddings, then the
    # output of each decoder layer.
    hidden = torch.stack([
        torch.stack([layer_h[0, -1] for layer_h in step_h])
        for step_h in out.hidden_states
    ])
    img = None
    if image_hidden:
        prefill = out.hidden_states[0]
        seq_len = prefill[0].shape[1]
        if seq_len != input_ids.shape[1] + N_IMAGE_PATCHES:
            raise SanityCheckError(
                f"prefill length {seq_len} != {input_ids.shape[1]} prompt tokens + "
                f"{N_IMAGE_PATCHES} patches; the image positions would be wrong")
        img = torch.stack([layer_h[0, 1:1 + N_IMAGE_PATCHES] for layer_h in prefill])
    return action, token_ids, hidden, img


def verify_patch_positions(model, inputs, img):
    """
    Layer 0 of the hidden states is the input embedding sequence, so at the
    image positions it must equal the projector's output for this image.
    If it doesn't, positions 1..256 are not the image patches.
    """
    import torch
    with torch.no_grad():
        proj = model.projector(model.vision_backbone(inputs["pixel_values"]))[0]
    if proj.shape != img[0].shape or not torch.allclose(proj, img[0], atol=1e-3, rtol=1e-3):
        diff = (proj.float() - img[0].float()).abs().max().item() if proj.shape == img[0].shape else None
        raise SanityCheckError(
            f"layer-0 hidden states at positions 1..{N_IMAGE_PATCHES} do not match the "
            f"projector output (shapes {tuple(proj.shape)} vs {tuple(img[0].shape)}, "
            f"max abs diff {diff})")


def patch_lens_for_step(img, norm, head, allowed, cand_ids_t, k=PATCH_TOPK):
    """
    Logit lens on the image patches: img [L+1, 256, d] through the final norm
    and output layer, at every layer, over the TEXT vocabulary only (`allowed`
    masks out special tokens, the 256 action bins and the padding rows).
    Same norm gotcha as lens_for_step: the last hidden state is already normed.

    Returns numpy arrays:
        top_ids    [L+1, 256, k]  int16    the k most likely words
        top_logp   [L+1, 256, k]  float16  their log-probabilities
        cand_rank  [L+1, 256, C]  uint16   rank of each candidate word (0 = top)
        cand_logp  [L+1, 256, C]  float16  its log-probability
    """
    import torch
    n_lay = img.shape[0]
    top_ids, top_logp, cand_rank, cand_logp = [], [], [], []
    with torch.no_grad():
        for l in range(n_lay):
            x = img[l] if l == n_lay - 1 else norm(img[l])
            logp = torch.log_softmax(
                head(x).float().masked_fill(~allowed, float("-inf")), dim=-1)  # [256, V]
            t = logp.topk(k, dim=-1)
            c = logp[:, cand_ids_t]                                            # [256, C]
            r = torch.stack([(logp > c[:, j:j + 1]).sum(-1) for j in range(c.shape[1])], dim=1)
            top_ids.append(t.indices.to(torch.int16))
            top_logp.append(t.values.half())
            cand_rank.append(r.to(torch.int32))
            cand_logp.append(c.half())
    stack = lambda xs: torch.stack(xs).cpu().numpy()
    return (stack(top_ids), stack(top_logp),
            stack(cand_rank).astype(np.uint16), stack(cand_logp))


def verify_against_predict_action(model, inputs, unnorm_key, action_ours):
    """Abort unless our generate() path reproduces predict_action exactly."""
    import torch
    with torch.no_grad():
        ref = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    if not np.array_equal(np.asarray(ref), np.asarray(action_ours)):
        raise SanityCheckError(
            "generate(output_hidden_states=True) action differs from predict_action:\n"
            f"  predict_action: {np.asarray(ref)}\n  ours:           {np.asarray(action_ours)}")


def lens_for_step(hidden, norm, head, ids_t, gen_token_ids, vocab):
    """
    Logit lens over all 7 action tokens x all L+1 layers for one env step.

    hidden: [7, L+1, d]. The GOTCHA: in HF Llama, hidden_states[0..L-1] are raw
    residual-stream states, but hidden_states[L] has ALREADY been through the
    final RMSNorm (modeling_llama applies self.norm before appending it). So we
    apply the norm to every layer except the last.

    Keeps only the 256 action-token logits, reordered so column b = bucket b,
    and softmaxes over those. Returns numpy arrays, each [7, L+1]:
        argmax   uint8     most likely bucket at that layer
        kl       float32   KL(P_final || P_layer), in nats
        entropy  float32   entropy of P_layer, in nats
    """
    import torch
    with torch.no_grad():
        x = torch.cat([norm(hidden[:, :-1]), hidden[:, -1:]], dim=1)
        full_logits = head(x).float()                          # [7, L+1, V]
        logits = full_logits[..., ids_t]                       # [7, L+1, 256]
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        # Break ties the way greedy generate() does: first max in TOKEN-ID
        # order. Columns are in bucket order, which is reversed token order, so
        # a plain argmax would pick the other side of an exact fp16 tie (run1
        # crashed on one: buckets 152 vs 154, margin 0.0).
        n_bins = logits.shape[-1]
        argmax = (n_bins - 1) - logits.flip(-1).argmax(dim=-1)
        logp_final = logp[:, -1:, :]
        kl = (logp_final.exp() * (logp_final - logp)).sum(-1)
        entropy = -(p * logp).sum(-1)

        # SANITY: the lens at the final layer IS the model's own output, so its
        # bucket must equal the token that generate() actually picked.
        gen_bucket = torch.as_tensor(vocab - 1 - gen_token_ids, device=argmax.device)
        final_bucket = argmax[:, -1]
        if not torch.equal(final_bucket, gen_bucket):
            bad = (final_bucket != gen_bucket).nonzero().flatten().tolist()
            top2 = logits[bad, -1].topk(2, dim=-1).values
            raise SanityCheckError(
                f"final-layer lens != generated token for action dims {bad}: "
                f"lens buckets {final_bucket[bad].tolist()}, generated "
                f"{gen_bucket[bad].tolist()}, lens top-2 logit margin "
                f"{(top2[:, 0] - top2[:, 1]).tolist()} (tiny margin = fp16 tie)")
        # Greedy decoding runs over the full vocab, so also check the model
        # never picked something outside the action range.
        full_argmax = full_logits[:, -1].argmax(-1).cpu().numpy()
        if not np.array_equal(full_argmax, gen_token_ids):
            raise SanityCheckError(
                f"full-vocab final-layer argmax {full_argmax} != generated {gen_token_ids}")

    return (argmax.to(torch.uint8).cpu().numpy(),
            kl.cpu().numpy().astype(np.float32),
            entropy.cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------------------
# Rollouts (runs A and B)
# ---------------------------------------------------------------------------

def run_episode(task, init_state, policy_fn, max_steps, on_step=None):
    """
    One episode, following run_libero_eval.py's loop:
      - NUM_STEPS_WAIT dummy actions while objects settle
      - then up to max_steps policy steps; stop early when the task is done
    policy_fn(img_224, step) -> raw 7-dim action (before gripper processing)
    on_step(obs, step, executed_action) lets --mode lens record per-step data. obs
    is the observation the model just SAW, so hidden states and simulator
    state line up row for row.
    Returns (success, num_policy_steps, frames).
    """
    env = make_env(task)
    try:
        env.reset()
        obs = env.set_init_state(init_state)
        t, step, done, frames = 0, 0, False, []
        while t < max_steps + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(DUMMY_ACTION)
                t += 1
                continue

            img = get_libero_image(obs)
            frames.append(img)
            action = postprocess_action(policy_fn(img, step))
            if on_step is not None:
                on_step(obs, step, action)

            try:
                obs, reward, done, info = env.step(action.tolist())
            except Exception as e:
                # Official behaviour: a simulator error ends the episode as a fail.
                print(f"[warn] env.step raised {type(e).__name__}: {e}; counting as fail")
                done = False
                step += 1
                break
            step += 1
            if done:
                break
            t += 1
        return bool(done), step, frames
    finally:
        env.close()


def read_episodes_csv(path):
    """{(task_id, init_idx): row dict} for whatever has already been written."""
    rows = {}
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(int(r["task_id"]), int(r["init_idx"]))] = r
    return rows


def append_episode_row(path, row):
    """Append one row and force it to disk, so a killed job loses nothing."""
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def print_success_summary(rows, tasks, inits):
    sel = [r for (t, i), r in rows.items() if t in tasks and i in inits]
    if not sel:
        print("[summary] no episodes recorded")
        return
    succ = [r["success"] in ("1", "True", True, 1) for r in sel]
    print(f"\n[summary] overall: {sum(succ)}/{len(sel)} = {np.mean(succ):.1%}")
    for t in tasks:
        ts = [r["success"] in ("1", "True", True, 1) for r in sel if int(r["task_id"]) == t]
        if ts:
            instr = next(r["instruction"] for r in sel if int(r["task_id"]) == t)
            print(f"  task {t:2d}: {sum(ts):2d}/{len(ts):2d} = {np.mean(ts):6.1%}  {instr}")


def run_rollouts(args):
    import torch
    from libero.libero import benchmark
    from singleVLA import (BASE_REPO, build_prompt, find_decoder_layers,
                           get_final_norm_and_head, load_model)

    get_tf()    # fail now, not after loading a 7B model, if TF is missing
    set_seed_everywhere(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = args.model_id or f"openvla/openvla-7b-finetuned-{args.suite.replace('_', '-')}"
    processor, model, dtype = load_model(model_id, BASE_REPO, device)

    # Same unnorm_key logic as run_libero_eval.py.
    unnorm_key = args.suite
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"{unnorm_key} not in {list(model.norm_stats)}"
    print(f"[info] unnorm_key = {unnorm_key}")

    record = args.mode in ("lens", "patchlens")
    patch = args.mode == "patchlens"
    if record:
        ids = action_token_ids(model, processor)
        ids_t = torch.as_tensor(ids, device=device)
        norm, head = get_final_norm_and_head(model)
        if norm is None:
            raise SanityCheckError("no final norm found; the logit lens needs it")
        n_layers = len(find_decoder_layers(model))
    if patch:
        # Text vocabulary = everything below the action bins, minus <unk> <s> </s>.
        allowed = torch.zeros(head.out_features, dtype=torch.bool, device=device)
        allowed[3:int(ids.min())] = True
        tok = processor.tokenizer
        cand_ids = [tok.convert_tokens_to_ids(t) for t in PATCH_CANDIDATES]
        bad = [t for t, i in zip(PATCH_CANDIDATES, cand_ids)
               if i is None or i == tok.unk_token_id or not allowed[i]]
        if bad:
            raise SanityCheckError(f"candidate words not single text tokens: {bad}")
        cand_ids_t = torch.as_tensor(cand_ids, device=device)
        print(f"[info] patch lens: {int(allowed.sum())} text tokens allowed, "
              f"{len(cand_ids)} candidate words {dict(zip(PATCH_CANDIDATES, cand_ids))}")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = [t for t in args.task_list if t < suite.n_tasks]
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    csv_path = os.path.join(args.out_dir, "episodes.csv")
    subs = ["videos"] + (["lens", "hidden"] if record else []) + \
           (["patchlens", "patch_hidden"] if patch else [])
    for sub in subs:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    rows = read_episodes_csv(csv_path)
    print(f"[info] {len(rows)} episodes already in {csv_path}")

    n_final_checks = 0
    t_start = time.time()
    for task_id in tasks:
        task = suite.get_task(task_id)
        instruction = task.language
        init_states = suite.get_task_init_states(task_id)
        prompt = build_prompt(instruction)

        for init_idx in args.init_list:
            key = (task_id, init_idx)
            stem = episode_stem(task_id, init_idx)
            lens_path = os.path.join(args.out_dir, "lens", stem + ".npz")
            hidden_path = os.path.join(args.out_dir, "hidden", stem + ".npz")
            patch_path = os.path.join(args.out_dir, "patchlens", stem + ".npz")
            patch_hidden_path = os.path.join(args.out_dir, "patch_hidden", stem + ".npz")
            needed = [lens_path, hidden_path] + ([patch_path, patch_hidden_path] if patch else [])

            # ---- resume: decide whether this episode is already done ----
            if record:
                if all(os.path.isfile(p) for p in needed):
                    if key not in rows:
                        # Crashed between writing the npz files and the CSV row.
                        meta = np.load(lens_path)
                        row = {"task_id": task_id, "init_idx": init_idx,
                               "instruction": instruction,
                               "success": int(meta["success"]),
                               "num_steps": int(meta["num_steps"])}
                        append_episode_row(csv_path, row)
                        rows[key] = {k: str(v) for k, v in row.items()}
                    continue
            elif key in rows:
                continue

            # ---- per-step recording for --mode lens ----
            rec = {"argmax": [], "kl": [], "entropy": [], "token_ids": [],
                   "hidden": [], "eef_pos": [], "gripper_qpos": [],
                   "action": [], "action_model": [], "objects": {},
                   "p_top_ids": [], "p_top_logp": [], "p_cand_rank": [], "p_cand_logp": [],
                   "p_image": [], "p_hidden": [], "p_hidden_steps": []}

            def policy_fn(img_224, step):
                nonlocal n_final_checks
                image = model_input_image(img_224)
                inputs = processor(prompt, image).to(device, dtype=dtype)
                with torch.no_grad():
                    if not record:
                        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                    action, token_ids, hidden, img = generate_with_hidden(
                        model, inputs, unnorm_key, image_hidden=patch)
                if hidden.shape[1] != n_layers + 1:
                    raise SanityCheckError(
                        f"expected {n_layers + 1} hidden states, got {hidden.shape[1]}")
                if step == 0:
                    # Check once per episode, on the first real observation.
                    verify_against_predict_action(model, inputs, unnorm_key, action)
                    if patch:
                        verify_patch_positions(model, inputs, img)
                if patch:
                    t_ids, t_lp, c_rank, c_lp = patch_lens_for_step(
                        img, norm, head, allowed, cand_ids_t)
                    rec["p_top_ids"].append(t_ids)
                    rec["p_top_logp"].append(t_lp)
                    rec["p_cand_rank"].append(c_rank)
                    rec["p_cand_logp"].append(c_lp)
                    rec["p_image"].append(np.asarray(image, dtype=np.uint8))
                    if step % PATCH_HIDDEN_EVERY == 0:
                        rec["p_hidden"].append(img.to(torch.float16).cpu().numpy())
                        rec["p_hidden_steps"].append(step)
                argmax, kl, ent = lens_for_step(hidden, norm, head, ids_t, token_ids,
                                                int(model.vocab_size))
                n_final_checks += len(token_ids)
                rec["argmax"].append(argmax)
                rec["kl"].append(kl)
                rec["entropy"].append(ent)
                rec["token_ids"].append(token_ids)
                # Position that produces the FIRST action token, every layer.
                rec["hidden"].append(hidden[0].to(torch.float16).cpu().numpy())
                rec["action_model"].append(np.asarray(action, dtype=np.float32))
                return action

            def on_step(obs, step, executed):
                if not record:
                    return
                rec["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                rec["gripper_qpos"].append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
                rec["action"].append(executed.astype(np.float32))
                for k in object_pos_keys(obs):
                    rec["objects"].setdefault(k, []).append(np.asarray(obs[k], dtype=np.float32))

            t_ep = time.time()
            success, num_steps, frames = run_episode(
                task, init_states[init_idx], policy_fn, max_steps, on_step)

            # ---- save outputs (npz first, CSV row last) ----
            if args.save_video:
                import imageio
                tag = "success" if success else "fail"
                imageio.mimsave(os.path.join(args.out_dir, "videos", f"{stem}_{tag}.mp4"),
                                frames, fps=30)
            if record:
                token_ids = np.stack(rec["token_ids"])
                save_npz_atomic(
                    lens_path,
                    argmax=np.stack(rec["argmax"]), kl=np.stack(rec["kl"]),
                    entropy=np.stack(rec["entropy"]), token_ids=token_ids,
                    buckets=(int(model.vocab_size) - 1 - token_ids).astype(np.uint8),
                    success=np.int8(success), num_steps=np.int32(num_steps),
                    task_id=np.int32(task_id), init_idx=np.int32(init_idx))
                save_npz_atomic(
                    hidden_path,
                    hidden=np.stack(rec["hidden"]),                  # [T, L+1, d] fp16
                    eef_pos=np.stack(rec["eef_pos"]),
                    gripper_qpos=np.stack(rec["gripper_qpos"]),
                    action=np.stack(rec["action"]),                  # executed
                    action_model=np.stack(rec["action_model"]),      # pre-gripper-processing
                    buckets=(int(model.vocab_size) - 1 - token_ids).astype(np.uint8),
                    task_id=np.int32(task_id), init_idx=np.int32(init_idx),
                    **{f"obj_{k}": np.stack(v) for k, v in rec["objects"].items()})
            if patch:
                save_npz_atomic(
                    patch_hidden_path,
                    hidden=np.stack(rec["p_hidden"]),                # [S, L+1, 256, d] fp16
                    steps=np.array(rec["p_hidden_steps"], dtype=np.int32),
                    task_id=np.int32(task_id), init_idx=np.int32(init_idx))
                save_npz_atomic(
                    patch_path,
                    top_ids=np.stack(rec["p_top_ids"]),              # [T, L+1, 256, k]
                    top_logp=np.stack(rec["p_top_logp"]),
                    cand_rank=np.stack(rec["p_cand_rank"]),          # [T, L+1, 256, C]
                    cand_logp=np.stack(rec["p_cand_logp"]),
                    cand_tokens=np.array(PATCH_CANDIDATES),
                    cand_ids=np.array(cand_ids, dtype=np.int32),
                    image=np.stack(rec["p_image"]),                  # [T, 224, 224, 3] model input
                    success=np.int8(success), num_steps=np.int32(num_steps),
                    task_id=np.int32(task_id), init_idx=np.int32(init_idx))

            row = {"task_id": task_id, "init_idx": init_idx, "instruction": instruction,
                   "success": int(success), "num_steps": num_steps}
            if key in rows:
                # A rollout-mode job in this out_dir already recorded this
                # episode. Don't duplicate the row,
                # but it must agree: this is a free determinism check.
                if int(rows[key]["success"]) != int(success):
                    print(f"[WARN] {stem}: success={int(success)} now but "
                          f"{rows[key]['success']} in episodes.csv. Episodes are "
                          "supposed to be deterministic; investigate.")
            else:
                append_episode_row(csv_path, row)
                rows[key] = {k: str(v) for k, v in row.items()}

            done_sel = [r for (t, i), r in rows.items() if t in tasks and i in args.init_list]
            n_succ = sum(int(r["success"]) for r in done_sel)
            print(f"[ep] task {task_id} init {init_idx}: "
                  f"{'SUCCESS' if success else 'fail   '} in {num_steps:3d} steps "
                  f"({time.time() - t_ep:.0f}s) | running {n_succ}/{len(done_sel)} = "
                  f"{n_succ / len(done_sel):.1%} | elapsed {(time.time() - t_start) / 60:.1f} min",
                  flush=True)

    if record:
        print(f"[ok] final-layer lens matched the generated token on all "
              f"{n_final_checks} action tokens this run")
    print_success_summary(rows, tasks, args.init_list)


# ---------------------------------------------------------------------------
# Plot styling (shared by --analyze and --mode probe)
# ---------------------------------------------------------------------------

# Fixed categorical order, one slot per action dim; never cycled.
DIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def new_axes(title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("layer (0 = embeddings, last = final post-norm)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    return fig, ax


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[plot] {path}")


def write_layer_csv(path, columns):
    """columns: {name: 1-D array over layers}. One row per layer."""
    names = list(columns)
    n = len(next(iter(columns.values())))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + names)
        for l in range(n):
            w.writerow([l] + [f"{columns[c][l]:.6g}" for c in names])
    print(f"[csv]  {path}")


# ---------------------------------------------------------------------------
# --mode lens --analyze: logit-lens plots (CPU only)
# ---------------------------------------------------------------------------

def analyze_lens(args):
    files = sorted(glob.glob(os.path.join(args.out_dir, "lens", "*.npz")))
    if not files:
        sys.exit(f"[error] no lens files in {args.out_dir}/lens; run --mode lens first")
    argmax, kl = [], []
    for fpath in files:
        d = np.load(fpath)
        argmax.append(d["argmax"])
        kl.append(d["kl"])
    argmax = np.concatenate(argmax).astype(np.int16)   # [N, 7, L+1]
    kl = np.concatenate(kl)
    n_steps, n_dims, n_lay = argmax.shape
    print(f"[info] {len(files)} episodes, {n_steps} env steps, {n_lay} layer readouts")

    final = argmax[:, :, -1:]
    agree = (argmax == final).mean(axis=0)              # [7, L+1]
    absdiff = np.abs(argmax - final).mean(axis=0)
    kl_mean = kl.mean(axis=0)
    layers = np.arange(n_lay)
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    def per_dim(ax, values):
        for k in range(n_dims):
            ax.plot(layers, values[k], color=DIM_COLORS[k], linewidth=1.6,
                    label=ACTION_NAMES[k])

    # 1. agreement with the final bucket
    fig, ax = new_axes("Logit lens: agreement with final action bucket",
                       "fraction of steps where layer argmax = final")
    per_dim(ax, agree)
    ax.plot(layers, agree.mean(axis=0), color=INK, linewidth=2.6, label="overall")
    ax.axhline(1 / 256, color=MUTED, linestyle=":", linewidth=1.2, label="chance (1/256)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    save_fig(fig, os.path.join(plots, "lens_agreement.png"))
    write_layer_csv(os.path.join(plots, "lens_agreement.csv"),
                    {"overall": agree.mean(axis=0),
                     **{ACTION_NAMES[k]: agree[k] for k in range(n_dims)},
                     "chance": np.full(n_lay, 1 / 256)})

    # 2. mean absolute bucket distance
    fig, ax = new_axes("Logit lens: distance from final action bucket",
                       "mean |bucket_layer - bucket_final|")
    per_dim(ax, absdiff)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_fig(fig, os.path.join(plots, "lens_bucket_distance.png"))
    write_layer_csv(os.path.join(plots, "lens_bucket_distance.csv"),
                    {ACTION_NAMES[k]: absdiff[k] for k in range(n_dims)})

    # 3. mean KL(P_final || P_layer)
    fig, ax = new_axes("Logit lens: KL(P_final || P_layer) over the 256 action tokens",
                       "mean KL (nats)")
    per_dim(ax, kl_mean)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_fig(fig, os.path.join(plots, "lens_kl.png"))
    write_layer_csv(os.path.join(plots, "lens_kl.csv"),
                    {ACTION_NAMES[k]: kl_mean[k] for k in range(n_dims)})


# ---------------------------------------------------------------------------
# --mode probe: ridge probes on hidden states (CPU only)
# ---------------------------------------------------------------------------

def load_hidden_dataset(out_dir):
    """
    Stack every hidden/*.npz. Returns
        H        [N, L+1, d] float16   all steps from all episodes
        groups   [N]                   episode index, for GroupKFold
        targets  {name: (Y [N, k], rows [N] bool, kind)}
    Object targets are only defined on episodes whose scene contains that
    object, which is what `rows` marks.
    """
    files = sorted(glob.glob(os.path.join(out_dir, "hidden", "*.npz")))
    if len(files) < 2:
        sys.exit(f"[error] need hidden states from >= 2 episodes in {out_dir}/hidden "
                 f"(found {len(files)}); probes are evaluated on held-out episodes")
    H, groups, per_ep = [], [], []
    for ep, fpath in enumerate(files):
        d = np.load(fpath)
        H.append(d["hidden"])
        groups.append(np.full(len(d["hidden"]), ep))
        per_ep.append({k: d[k] for k in d.files if k != "hidden"})
    H = np.concatenate(H)
    groups = np.concatenate(groups)
    lengths = [len(g) for g in np.split(groups, np.flatnonzero(np.diff(groups)) + 1)]
    print(f"[info] {len(files)} episodes, {len(H)} steps, hidden {H.shape[1:]} "
          f"({H.nbytes / 1e9:.1f} GB in RAM)")

    def gather(key, width):
        Y = np.full((len(H), width), np.nan, dtype=np.float64)
        rows = np.zeros(len(H), dtype=bool)
        start = 0
        for ep, n in enumerate(lengths):
            if key in per_ep[ep]:
                Y[start:start + n] = per_ep[ep][key].reshape(n, -1)
                rows[start:start + n] = True
            start += n
        return Y, rows

    targets = {}
    Y, rows = gather("eef_pos", 3)
    targets["eef_pos"] = (Y, rows, "perception")
    Y, rows = gather("gripper_qpos", 2)
    targets["gripper_qpos"] = (Y, rows, "perception")
    obj_keys = sorted({k for e in per_ep for k in e if k.startswith("obj_")})
    for k in obj_keys:
        Y, rows = gather(k, 3)
        targets[k[len("obj_"):]] = (Y, rows, "perception")
    Y, rows = gather("action", 7)
    for j, name in enumerate(ACTION_NAMES):
        targets[f"action_{name}"] = (Y[:, j:j + 1], rows, "action")
    return H, groups, targets


def run_probes(args):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    H, groups, targets = load_hidden_dataset(args.src_dir or args.out_dir)
    n_layers = H.shape[1]
    alphas = np.logspace(-1, 5, 13)

    # Drop coordinates that never change (R^2 is undefined for a constant),
    # e.g. an object that sits in the same place in every init state.
    cols = {}   # target -> indices of its non-constant coordinates
    for name, (Y, rows, _) in targets.items():
        keep = [j for j in range(Y.shape[1]) if np.nanstd(Y[rows, j]) > 1e-4]
        if keep:
            cols[name] = keep
        else:
            print(f"[warn] {name}: constant across all steps, skipped")

    # Targets defined on the same rows are fit together. Ridge with
    # alpha_per_target=True fits each output column independently with its own
    # alpha, so this is exactly "one target at a time", just batched.
    batches = {}
    for name in cols:
        rows = targets[name][1]
        batches.setdefault(rows.tobytes(), (rows, []))[1].append(name)

    r2 = {name: np.full(n_layers, np.nan) for name in cols}
    t0 = time.time()
    for rows, names in batches.values():
        idx = np.flatnonzero(rows)
        Y = np.concatenate([targets[n][0][idx][:, cols[n]] for n in names], axis=1)
        spans, s = [], 0
        for n in names:
            spans.append((n, slice(s, s + len(cols[n]))))
            s += len(cols[n])
        g = groups[idx]
        n_splits = min(5, len(np.unique(g)))
        # GroupKFold: every step of an episode lands in the same fold, so the
        # probe is always scored on episodes it never saw during training.
        folds = list(GroupKFold(n_splits=n_splits).split(idx, groups=g))
        print(f"[probe] {len(names)} targets on {len(idx)} steps, "
              f"{len(np.unique(g))} episodes, {n_splits}-fold GroupKFold")

        for layer in range(0, n_layers, args.layer_stride):
            X = H[idx, layer].astype(np.float32)
            pred = np.zeros_like(Y)
            for train, test in folds:
                # Standardize with TRAIN statistics only.
                mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
                model = RidgeCV(alphas=alphas, alpha_per_target=True)
                model.fit((X[train] - mu) / sd, Y[train])
                pred[test] = model.predict((X[test] - mu) / sd).reshape(len(test), -1)
            # Held-out R^2, pooled over folds, averaged over a target's coordinates.
            ss_res = ((Y - pred) ** 2).sum(0)
            ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
            r2_col = 1 - ss_res / ss_tot
            for n, sl in spans:
                r2[n][layer] = r2_col[sl].mean()
            print(f"  layer {layer:2d} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    write_layer_csv(os.path.join(plots, "probe_r2.csv"), r2)

    # Onset layer: first layer reaching 80% of that target's best R^2.
    print("\n[onset] first layer where held-out R^2 >= 80% of its max")
    with open(os.path.join(plots, "probe_onset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "kind", "max_r2", "max_layer", "onset_layer"])
        for name, vals in r2.items():
            kind = targets[name][2]
            if np.all(np.isnan(vals)) or np.nanmax(vals) <= 0:
                print(f"  {name:40s} {kind:10s} max R^2 <= 0, no onset")
                w.writerow([name, kind, f"{np.nanmax(vals):.4f}", "", ""])
                continue
            best = np.nanmax(vals)
            onset = int(np.flatnonzero(np.nan_to_num(vals, nan=-np.inf) >= 0.8 * best)[0])
            print(f"  {name:40s} {kind:10s} max R^2 {best:.3f} @ layer "
                  f"{int(np.nanargmax(vals)):2d}  onset layer {onset:2d}")
            w.writerow([name, kind, f"{best:.4f}", int(np.nanargmax(vals)), onset])

    # Plot: perception in blues (solid), actions in oranges (dashed), so the two
    # families stay distinguishable without relying on color alone.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    percep = [n for n in r2 if targets[n][2] == "perception"]
    action = [n for n in r2 if targets[n][2] == "action"]
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, max(len(percep), 1)))
    oranges = plt.get_cmap("Oranges")(np.linspace(0.45, 0.95, max(len(action), 1)))
    fig, ax = new_axes("Linear probes: held-out R² by layer (GroupKFold by episode)",
                       "held-out R²")
    layers = np.arange(n_layers)
    for c, n in zip(blues, percep):
        m = ~np.isnan(r2[n])
        ax.plot(layers[m], r2[n][m], color=c, linewidth=1.6, label=n)
    for c, n in zip(oranges, action):
        m = ~np.isnan(r2[n])
        ax.plot(layers[m], r2[n][m], color=c, linewidth=1.6, linestyle="--", label=n)
    # Clip the axis at -0.5: a very negative R^2 just means "no linear signal"
    # and would squash the interesting range. probe_r2.csv has the true values.
    ax.set_ylim(-0.5, 1.02)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.set_size_inches(11, 5)
    save_fig(fig, os.path.join(plots, "probe_r2.png"))


# ---------------------------------------------------------------------------
# --mode tunedlens: per-layer affine translators on saved hidden states (GPU)
# ---------------------------------------------------------------------------
#
# Tuned lens (Belrose et al. 2023, arXiv:2303.08112) on the first-action-token
# position (the one that produces x), using the hidden states --mode lens saved.
# For every layer l < L, train a translator A_l x + b_l in front of the FROZEN
# final RMSNorm and the 256 action rows of the FROZEN lm_head, to match the
# model's own final distribution over the 256 buckets: KL(P_final || P_lens).
# Nothing in OpenVLA is trained. A_l starts at the identity, so before training
# the tuned lens IS the logit lens.
#
# Translator variants (capacity ladder; read conclusions off what is stable):
#   bias         A = I, only b_l
#   r8/r32/r128  A = I + U V^T, rank 8/32/128 (V starts at 0)
#   full         A = I + M, M a full d x d matrix (starts at 0)
#   r32_shuffled r32 trained on targets permuted across steps: a control that
#                should score at the most-common-bucket baseline
#
# Evaluation is always on held-out episodes (k folds by episode, out-of-fold
# predictions pooled), with early stopping on a validation slice of the
# training episodes.

TL_VARIANTS = "bias,r8,r32,r128,full,r32_shuffled"
TL_MAIN_VARIANT = "r32"            # the headline curve in the gap plot


def episode_folds(groups, k):
    """Round-robin episodes into k folds. Returns [(train_idx, test_idx)]."""
    eps = np.unique(groups)
    fold_of = {e: i % k for i, e in enumerate(eps)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def split_val(train_idx, groups, every=8):
    """Every `every`-th training episode becomes the early-stopping validation set."""
    eps = np.unique(groups[train_idx])
    val_eps = set(eps[::every])
    is_val = np.array([groups[i] in val_eps for i in train_idx])
    return train_idx[~is_val], train_idx[is_val]


def bucket_argmax(logits):
    """
    Argmax over buckets with greedy generate()'s tie-break: first max in
    TOKEN-ID order, i.e. the HIGHEST bucket among exact ties (see lens_for_step).
    """
    n = logits.shape[-1]
    return (n - 1) - logits.flip(-1).argmax(dim=-1)


def load_tunedlens_data(src_dir):
    """
    hidden/*.npz from a --mode lens run. Returns
        H        [N, L+1, d] float16  first-action-token position, all layers
        final    [N] int              bucket the model actually generated for x
        groups   [N] int              episode index
        lens_am  [N, L+1] int or None the recorded logit-lens argmax for x
    """
    files = sorted(glob.glob(os.path.join(src_dir, "hidden", "*.npz")))
    if len(files) < 2:
        sys.exit(f"[error] need hidden states from >= 2 episodes in {src_dir}/hidden "
                 f"(found {len(files)}); run --mode lens first")
    H, final, groups, lens_am = [], [], [], []
    for ep, fpath in enumerate(files):
        d = np.load(fpath)
        H.append(d["hidden"])
        final.append(d["buckets"][:, 0].astype(np.int64))
        groups.append(np.full(len(d["hidden"]), ep))
        lpath = os.path.join(src_dir, "lens", os.path.basename(fpath))
        if lens_am is not None and os.path.isfile(lpath):
            lens_am.append(np.load(lpath)["argmax"][:, 0, :].astype(np.int64))
        else:
            lens_am = None
    H = np.concatenate(H)
    print(f"[info] {len(files)} episodes, {len(H)} steps, hidden {H.shape[1:]} "
          f"({H.nbytes / 1e9:.1f} GB in RAM)")
    return (H, np.concatenate(final), np.concatenate(groups),
            None if lens_am is None else np.concatenate(lens_am))


def tunedlens_readout(args, device):
    """
    The frozen read-out the lens needs: the 256 action rows of lm_head (row b =
    bucket b) and the final RMSNorm. Pulled from the model once and cached in
    out_dir/tunedlens/readout.npz (about 4 MB), so reruns skip the model load.
    """
    import torch
    path = os.path.join(args.out_dir, "tunedlens", "readout.npz")
    if not os.path.isfile(path):
        from singleVLA import BASE_REPO, get_final_norm_and_head, load_model
        model_id = args.model_id or f"openvla/openvla-7b-finetuned-{args.suite.replace('_', '-')}"
        processor, model, _ = load_model(model_id, BASE_REPO, device)
        ids = action_token_ids(model, processor)
        norm, head = get_final_norm_and_head(model)
        if norm is None:
            raise SanityCheckError("no final norm found; the tuned lens needs it")
        eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
        if eps is None:
            raise SanityCheckError(f"can't find the epsilon of {type(norm).__name__}")
        ids_t = torch.as_tensor(ids, device=head.weight.device)
        save_npz_atomic(path,
                        W_act=head.weight[ids_t].detach().float().cpu().numpy(),   # [256, d]
                        norm_w=norm.weight.detach().float().cpu().numpy(),         # [d]
                        eps=np.float32(eps), model_id=np.array(model_id))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[info] cached read-out from {model_id} in {path}")
    r = np.load(path)
    print(f"[info] read-out <- {path} ({str(r['model_id'])})")
    return (torch.as_tensor(r["W_act"], device=device),
            torch.as_tensor(r["norm_w"], device=device), float(r["eps"]))


def make_translator(kind, d):
    import torch
    import torch.nn as nn

    class Translator(nn.Module):
        """A x + b with A = I + (low-rank or full) delta. Starts as the identity."""
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(d))
            self.U = self.V = self.M = None
            if kind.startswith("r"):
                rank = int(kind[1:].split("_")[0])
                self.U = nn.Parameter(torch.randn(d, rank) / d ** 0.5)
                self.V = nn.Parameter(torch.zeros(rank, d))
            elif kind == "full":
                self.M = nn.Parameter(torch.zeros(d, d))
            elif kind != "bias":
                raise ValueError(f"unknown translator variant {kind!r}")

        def forward(self, x):
            out = x + self.b
            if self.U is not None:
                out = out + (x @ self.U) @ self.V
            if self.M is not None:
                out = out + x @ self.M
            return out

    return Translator()


def run_tunedlens(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = args.src_dir or args.out_dir
    out = os.path.join(args.out_dir, "tunedlens")
    os.makedirs(out, exist_ok=True)
    set_seed_everywhere(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False      # keep the fp32 lens exact

    H, final, groups, lens_am = load_tunedlens_data(src)
    W, norm_w, eps = tunedlens_readout(args, device)
    n, n_lay, d = H.shape
    L = n_lay - 1
    final_t = torch.as_tensor(final, device=device)
    variants = [v.strip() for v in args.tl_variants.split(",") if v.strip()]
    layers = list(range(0, L, args.layer_stride))

    def rmsnorm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w

    def logp_of(x_normed):
        return torch.log_softmax(x_normed @ W.T, dim=-1)

    def batched(fn, X, bs=2048):
        return torch.cat([fn(X[i:i + bs]) for i in range(0, len(X), bs)])

    # ---- targets: the model's own final distribution over the 256 buckets ----
    # hidden_states[L] is ALREADY through the final RMSNorm (see lens_for_step).
    with torch.no_grad():
        logp_final = batched(logp_of, torch.as_tensor(H[:, L], device=device).float())
    fb = bucket_argmax(logp_final)
    mismatch = int((fb != final_t).sum())
    print(f"[check] fp32 final-layer bucket == generated bucket on "
          f"{n - mismatch}/{n} steps")
    if mismatch > max(1, n // 200):
        raise SanityCheckError(
            f"fp32 read-out disagrees with the generated x bucket on {mismatch}/{n} "
            "steps; the cached read-out or the saved position is wrong")

    # SANITY: layer 0 at this position is the embedding of the "▁" token that
    # ends every prompt, so it must be identical at every step.
    h0 = torch.as_tensor(H[:, 0], device=device).float()
    if float((h0 - h0[0]).abs().max()) > 1e-3:
        raise SanityCheckError(
            "layer-0 hidden state varies across steps; this is not the "
            "first-action-token position")
    del h0

    def metrics(lp, sel=None):
        """lp: [m, 256] lens log-probs for the steps in sel."""
        sel = slice(None) if sel is None else sel
        p_f, f = logp_final[sel], final_t[sel]
        am = bucket_argmax(lp)
        dist = (am - f).abs().float()
        return {"agree": (am == f).float().mean().item(),
                "within5": (dist <= 5).float().mean().item(),
                "within25": (dist <= 25).float().mean().item(),
                "dist": dist.mean().item(),
                "kl": (p_f.exp() * (p_f - lp)).sum(-1).mean().item(),
                "entropy": -(lp.exp() * lp).sum(-1).mean().item()}, am

    names = ["agree", "within5", "within25", "dist", "kl", "entropy"]

    # ---- logit lens (identity translator), every layer, no training ----
    logit = {k: np.zeros(n_lay) for k in names}
    with torch.no_grad():
        for l in range(n_lay):
            X = torch.as_tensor(H[:, l], device=device).float()
            lp = batched(lambda x: logp_of(x if l == L else rmsnorm(x)), X)
            m, am = metrics(lp)
            for k in names:
                logit[k][l] = m[k]
            if lens_am is not None and l < L:
                same = (am.cpu().numpy() == lens_am[:, l]).mean()
                if same < 0.95:
                    print(f"[warn] layer {l}: recomputed logit-lens argmax matches the "
                          f"recorded one on only {same:.1%} of steps")
    print(f"[info] logit lens agreement: layer 0 {logit['agree'][0]:.3f}, "
          f"layer {L // 2} {logit['agree'][L // 2]:.3f}, layer {L - 1} {logit['agree'][L - 1]:.3f}")

    # ---- baselines from the training folds, scored out of fold ----
    folds = episode_folds(groups, args.tl_folds)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "fold leaks episodes"
    base_pred = np.zeros(n, dtype=np.int64)
    base_lp = torch.zeros_like(logp_final)
    for tr, te in folds:
        base_pred[te] = np.bincount(final[tr], minlength=256).argmax()
        marg = logp_final[tr].exp().mean(0).clamp_min(1e-12)
        base_lp[te] = (marg / marg.sum()).log()
    baseline_agree = float((base_pred == final).mean())
    baseline_kl = metrics(base_lp)[0]["kl"]
    print(f"[info] most-common-bucket baseline (out of fold): agreement {baseline_agree:.3f}, "
          f"KL of the marginal {baseline_kl:.3f} nats")

    # ---- train translators ----
    res = {k: np.full((len(variants), len(layers)), np.nan) for k in names}
    oof_am = np.zeros((n, len(variants), len(layers)), dtype=np.uint8)
    epochs_used = np.zeros((len(variants), len(layers), len(folds)), dtype=np.int32)
    gen = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    for li, l in enumerate(layers):
        X_all = torch.as_tensor(H[:, l], device=device).float()
        for vi, kind in enumerate(variants):
            shuffled = kind.endswith("_shuffled")
            lp_oof = torch.zeros_like(logp_final)
            for fi, (tr, te) in enumerate(folds):
                tr_fit, va = split_val(tr, groups)
                # One constant scale per layer from the training steps keeps the
                # map affine while putting every layer on a similar scale; the
                # final RMSNorm is scale-invariant, so the identity is unchanged.
                s = float(torch.rsqrt(X_all[tr].pow(2).mean()))
                Xs = X_all * s
                y_fit, y_va = logp_final[tr_fit], logp_final[va]
                if shuffled:
                    y_fit = y_fit[torch.randperm(len(tr_fit), generator=gen).to(device)]
                    y_va = y_va[torch.randperm(len(va), generator=gen).to(device)]
                T = make_translator(kind, d).to(device)
                # Weight decay on the delta pulls A back toward the identity.
                decay = [p for nm, p in T.named_parameters() if nm != "b"]
                groups_opt = [{"params": [T.b], "weight_decay": 0.0}]
                if decay:
                    groups_opt.append({"params": decay, "weight_decay": args.tl_wd})
                opt = torch.optim.AdamW(groups_opt, lr=args.tl_lr)

                def kl(lp_target, x):
                    lq = logp_of(rmsnorm(T(x)))
                    return (lp_target.exp() * (lp_target - lq)).sum(-1).mean()

                Xf, Xv = Xs[tr_fit], Xs[va]
                best, best_state, bad, ep = float("inf"), None, 0, 0
                for ep in range(1, args.tl_epochs + 1):
                    perm = torch.randperm(len(tr_fit), generator=gen).to(device)
                    for i in range(0, len(perm), args.tl_batch):
                        idx = perm[i:i + args.tl_batch]
                        loss = kl(y_fit[idx], Xf[idx])
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        v = kl(y_va, Xv).item()
                    if v < best - 1e-4:
                        best, bad = v, 0
                        best_state = {k: t.detach().clone() for k, t in T.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= args.tl_patience:
                            break
                T.load_state_dict(best_state)
                epochs_used[vi, li, fi] = ep
                with torch.no_grad():
                    lp_oof[te] = batched(lambda x: logp_of(rmsnorm(T(x))), Xs[te])
                del T, opt, best_state
            m, am = metrics(lp_oof)
            for k in names:
                res[k][vi, li] = m[k]
            oof_am[:, vi, li] = am.cpu().numpy().astype(np.uint8)
        row = "  ".join(f"{v} {res['agree'][vi, li]:.3f}" for vi, v in enumerate(variants))
        print(f"[layer {l:2d}] agreement: logit {logit['agree'][l]:.3f}  {row}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        del X_all

    # SANITY: at layer 0 every step has the same input, so no translator can
    # beat the most-common-bucket baseline on held-out episodes.
    if 0 in layers:
        worst = float(np.nanmax(res["agree"][:, layers.index(0)]))
        if worst > baseline_agree + 0.02:
            raise SanityCheckError(
                f"a translator scores {worst:.3f} at layer 0, above the baseline "
                f"{baseline_agree:.3f}; something leaks across folds")

    save_npz_atomic(os.path.join(out, "results.npz"),
                    variants=np.array(variants), layers=np.array(layers),
                    **{f"tl_{k}": v for k, v in res.items()},
                    **{f"logit_{k}": v for k, v in logit.items()},
                    baseline_agree=np.float64(baseline_agree),
                    baseline_kl=np.float64(baseline_kl),
                    oof_argmax=oof_am, final=final.astype(np.uint8),
                    groups=groups.astype(np.int32), epochs_used=epochs_used,
                    n_layers=np.int32(L), src_dir=np.array(src))
    print(f"[ok] translators trained on {len(layers)} layers x {len(variants)} variants "
          f"x {len(folds)} folds in {(time.time() - t0) / 60:.1f} min")
    plot_tunedlens(args.out_dir)


def onset(values, layers, level):
    """First layer whose value reaches `level`, or None."""
    for l, v in zip(layers, values):
        if np.isfinite(v) and v >= level:
            return int(l)
    return None


def plot_tunedlens(out_dir):
    """Plots and CSVs from tunedlens/results.npz (CPU only), plus the gap plot."""
    r = np.load(os.path.join(out_dir, "tunedlens", "results.npz"))
    variants = [str(v) for v in r["variants"]]
    layers = r["layers"]
    L = int(r["n_layers"])
    base = float(r["baseline_agree"])
    all_layers = np.arange(L + 1)
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    def with_final(vals, final_value):
        """Translator values at `layers`, plus the final layer (no lens needed)."""
        full = np.full(L + 1, np.nan)
        full[layers] = vals
        full[L] = final_value
        return full

    finals = {"agree": 1.0, "within5": 1.0, "within25": 1.0, "dist": 0.0, "kl": 0.0}
    for metric in ["agree", "within5", "within25", "dist", "kl"]:
        cols = {"logit_lens": r[f"logit_{metric}"]}
        for vi, v in enumerate(variants):
            cols[v] = with_final(r[f"tl_{metric}"][vi], finals[metric])
        if metric == "agree":
            cols["baseline"] = np.full(L + 1, base)
        if metric == "kl":
            cols["baseline_marginal"] = np.full(L + 1, float(r["baseline_kl"]))
        write_layer_csv(os.path.join(plots, f"tunedlens_{metric}.csv"), cols)

    # Translator variants get one ordered warm ramp (capacity), the logit lens
    # stays neutral, the shuffled control is dotted.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real = [v for v in variants if not v.endswith("_shuffled")]
    ramp = dict(zip(real, plt.get_cmap("Oranges")(np.linspace(0.4, 0.95, max(len(real), 1)))))

    def lines(ax, metric):
        ax.plot(all_layers, r[f"logit_{metric}"], color=MUTED, linewidth=2.2, label="logit lens")
        for vi, v in enumerate(variants):
            y = with_final(r[f"tl_{metric}"][vi], finals[metric])
            m = np.isfinite(y)
            if v.endswith("_shuffled"):
                ax.plot(all_layers[m], y[m], color=INK, linestyle=":", linewidth=1.4,
                        label=f"{v} (control)")
            else:
                ax.plot(all_layers[m], y[m], color=ramp[v], linewidth=1.8,
                        label=f"tuned lens, {v}")

    fig, ax = new_axes("Tuned vs logit lens for x: agreement with the final bucket "
                       "(held-out episodes)", "fraction of steps where lens argmax = final")
    lines(ax, "agree")
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2,
               label=f"most-common bucket ({base:.2f})")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, os.path.join(plots, "tunedlens_agreement.png"))

    fig, ax = new_axes("Tuned vs logit lens for x: KL(P_final || P_lens) (held-out episodes)",
                       "mean KL (nats)")
    lines(ax, "kl")
    ax.axhline(float(r["baseline_kl"]), color=MUTED, linestyle="--", linewidth=1.2,
               label="marginal distribution")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, os.path.join(plots, "tunedlens_kl.png"))

    # ---- summary: when does each lens first read the action? ----
    # gain = (agreement - baseline) / (1 - baseline): 0 = baseline, 1 = always right.
    def gain(a):
        return (np.asarray(a) - base) / (1 - base)

    final = r["final"].astype(np.int64)
    rows = []
    curves = {"logit_lens": (all_layers, r["logit_agree"])}
    for vi, v in enumerate(variants):
        curves[v] = (np.append(layers, L), np.append(r["tl_agree"][vi], 1.0))
    for name, (ls, a) in curves.items():
        settle = ""
        if name != "logit_lens" and np.array_equal(layers, np.arange(L)):
            # Settle layer: first layer from which every later layer picks the final bucket.
            am = r["oof_argmax"][:, variants.index(name), :].astype(np.int64)
            ok = np.concatenate([am == final[:, None], np.ones((len(final), 1), bool)], axis=1)
            settle_l = L - np.cumprod(ok[:, ::-1], axis=1).sum(1) + 1
            settle = f"{np.median(settle_l):.0f}"
        rows.append([name, f"{a[0]:.4f}", onset(a, ls, base + 0.05),
                     onset(gain(a), ls, 0.5), settle])
    summary = os.path.join(plots, "tunedlens_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lens", "layer0_agreement", "first_layer_beats_baseline_by_5pts",
                    "first_layer_half_way_to_perfect", "median_settle_layer"])
        w.writerows(rows)
    print(f"[csv]  {summary}  (baseline {base:.4f})")
    for row in rows:
        print(f"  {row[0]:14s} layer0 {row[1]}  beats baseline+5pts at {row[2]}  "
              f"half-way at {row[3]}  median settle {row[4] or '-'}")

    plot_gap(out_dir, curves, base, L)


def plot_gap(out_dir, curves, base, L):
    """
    Perception vs action on ONE residual stream (the first-action-token
    position): ridge-probe R^2 for scene state (from --mode probe, same out_dir)
    against the tuned lens's gain over the baseline. Both are on a 0..1 scale.
    """
    plots = os.path.join(out_dir, "plots")
    probe_csv = os.path.join(plots, "probe_r2.csv")
    if not os.path.isfile(probe_csv):
        print(f"[info] no {probe_csv}; run --mode probe with this --out_dir for the gap plot")
        return
    with open(probe_csv) as f:
        rd = list(csv.DictReader(f))
    p_layers = np.array([int(x["layer"]) for x in rd])
    probes = {k: np.array([float(x[k]) if x[k] not in ("", "nan") else np.nan for x in rd])
              for k in rd[0] if k != "layer"}
    percep = {k: v for k, v in probes.items() if not k.startswith("action_")}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = new_axes("Perception vs action at the first-action-token position",
                       "probe R²  /  tuned-lens gain over baseline")
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, max(len(percep), 1)))
    p_onsets = []
    for c, (k, v) in zip(blues, percep.items()):
        m = np.isfinite(v)
        ax.plot(p_layers[m], v[m], color=c, linewidth=1.4, label=f"probe: {k}")
        if np.nanmax(v) > 0:
            p_onsets.append(onset(v, p_layers, 0.8 * np.nanmax(v)))
    if "action_x" in probes:
        v = probes["action_x"]
        m = np.isfinite(v)
        ax.plot(p_layers[m], v[m], color="#eb6834", linestyle="--", linewidth=1.4,
                label="probe: executed x (ground truth)")
    for name, color, style in [("logit_lens", MUTED, "-"), (TL_MAIN_VARIANT, "#b23a0e", "-"),
                               ("full", "#eda100", "-")]:
        if name in curves:
            ls, a = curves[name]
            g = (np.asarray(a) - base) / (1 - base)
            ax.plot(ls, g, color=color, linestyle=style, linewidth=2.4,
                    label=("logit lens" if name == "logit_lens" else f"tuned lens, {name}") + " gain")
    p_onsets = [o for o in p_onsets if o is not None]
    P = int(np.median(p_onsets)) if p_onsets else None
    A = None
    if TL_MAIN_VARIANT in curves:
        ls, a = curves[TL_MAIN_VARIANT]
        A = onset((np.asarray(a) - base) / (1 - base), ls, 0.5)
    # P and A get their own colour and height so they stay readable when equal.
    for x, lab, color, y in [(P, "P (perception)", "#08519c", 0.98),
                             (A, "A (action)", "#b23a0e", 0.88)]:
        if x is not None:
            ax.axvline(x, color=color, linestyle=":", linewidth=1.2)
            ax.text(x + 0.3, y, lab, color=color, fontsize=9, va="top")
    ax.set_ylim(-0.3, 1.02)
    ax.set_xlim(0, L)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.set_size_inches(11, 5)
    save_fig(fig, os.path.join(plots, "gap_perception_action.png"))

    path = os.path.join(plots, "gap_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "layer", "definition"])
        w.writerow(["P", "" if P is None else P,
                    "median over perception probes of the first layer reaching 80% of that probe's max R^2"])
        w.writerow(["A", "" if A is None else A,
                    f"first layer where the {TL_MAIN_VARIANT} tuned lens is half-way from the "
                    "most-common-bucket baseline to perfect agreement"])
        w.writerow(["gap", "" if P is None or A is None else A - P, "A - P"])
    print(f"[csv]  {path}  P={P} A={A}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["rollout", "lens", "probe", "patchlens", "tunedlens"],
                   required=True)
    p.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    p.add_argument("--tasks", default="0-9", help='task ids, e.g. "0-9" or "0,3,5"')
    p.add_argument("--inits", default="0-4", help='init state indices, e.g. "0-4"')
    p.add_argument("--out_dir", required=True,
                   help="this run's folder, e.g. runs/run1")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--src_dir", default=None,
                   help="--mode probe/tunedlens: read hidden/ (and lens/) from this run's "
                        "folder instead of --out_dir, e.g. runs/run2")
    p.add_argument("--analyze", action="store_true",
                   help="with --mode lens: plot saved lens data, no model load; with "
                        "--mode tunedlens: replot saved translator results")
    p.add_argument("--dry_run", action="store_true",
                   help="1 task, 1 init, 20 steps")
    p.add_argument("--model_id", default=None,
                   help="default: openvla/openvla-7b-finetuned-<suite>")
    p.add_argument("--max_steps", type=int, default=None,
                   help="override the official per-suite step budget")
    p.add_argument("--layer_stride", type=int, default=1,
                   help="--mode probe/tunedlens: every Nth layer (1 = all)")
    p.add_argument("--tl_variants", default=TL_VARIANTS,
                   help="--mode tunedlens: comma-separated translator variants")
    p.add_argument("--tl_folds", type=int, default=5,
                   help="--mode tunedlens: cross-validation folds (by episode)")
    p.add_argument("--tl_epochs", type=int, default=60,
                   help="--mode tunedlens: max epochs per translator")
    p.add_argument("--tl_patience", type=int, default=5,
                   help="--mode tunedlens: early-stopping patience in epochs")
    p.add_argument("--tl_lr", type=float, default=1e-3)
    p.add_argument("--tl_wd", type=float, default=1e-2,
                   help="--mode tunedlens: weight decay on the translator delta")
    p.add_argument("--tl_batch", type=int, default=512)
    args = p.parse_args()

    args.task_list = parse_range(args.tasks)
    args.init_list = parse_range(args.inits)
    if args.dry_run and args.mode in ("rollout", "lens", "patchlens") and not args.analyze:
        args.task_list, args.init_list, args.max_steps = [0], [0], 20
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] mode {args.mode}{' --analyze' if args.analyze else ''} | "
          f"suite {args.suite} | out_dir {args.out_dir}")

    if args.mode == "probe":
        run_probes(args)
    elif args.mode == "tunedlens":
        plot_tunedlens(args.out_dir) if args.analyze else run_tunedlens(args)
    elif args.mode == "lens" and args.analyze:
        analyze_lens(args)
    else:
        print(f"[info] tasks {args.task_list} | inits {args.init_list}")
        run_rollouts(args)


if __name__ == "__main__":
    main()
