"""
run_libero_single_interp.py

Interpretability-oriented version of the OpenVLA + LIBERO smoke test.

Adds, on top of the original single-episode rollout:
  - forward hooks on every decoder layer of the language backbone, capturing
    the LAST-TOKEN hidden state at every generation step (the prefill pass
    over the prompt+image tokens, plus each autoregressive action-token
    step after it)
  - organizes the result per environment timestep as a tensor of shape
        [n_gen_steps, n_layers, d_model]
    where n_gen_steps ~= 1 (prefill) + n_action_tokens (7 for a standard
    7-DoF action: dx, dy, dz, droll, dpitch, dyaw, gripper) -- this mirrors
    how OpenVLA actually decodes one action token per autoregressive step.
  - a logit-lens helper that projects any layer's hidden state through the
    model's OWN final norm + LM head, skipping every later layer, so you can
    see what token / action-bin that layer "would have predicted" on its own.

IMPORTANT -- verify before trusting this on real data:
  OpenVLA/Prismatic's exact module nesting (where the decoder layer list
  lives, what the final norm/lm_head are called) can differ between forks
  and checkpoints. Run with `--inspect` FIRST:

      python run_libero_single_interp.py --inspect

  This loads the model, prints its module tree, and exits without touching
  LIBERO. Confirm (or fix) the attribute paths in `find_decoder_layers()`
  and `get_final_norm_and_head()` below against what you see before running
  a real episode -- a silently-wrong hook path will record garbage, not an
  error.

Usage (after --inspect confirms the paths are right):
    python run_libero_single_interp.py --task_suite libero_spatial --task_id 0
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------------
# Model-introspection helpers -- VERIFY THESE against your actual checkpoint
# via --inspect before trusting any saved hidden states.
# ---------------------------------------------------------------------------

def find_decoder_layers(model):
    """
    Locate 'the list of transformer decoder layers' inside the VLA. Tries a
    few attribute paths seen across OpenVLA/Prismatic forks; raises loudly
    if none match rather than silently hooking nothing.
    """
    candidates = [
        "language_model.model.layers",     # typical HF causal-LM nesting
        "llm_backbone.llm.model.layers",   # some Prismatic-vlms trees
        "model.language_model.model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                print(f"[info] found decoder layers at model.{path} ({len(obj)} layers)")
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not find the decoder layer list automatically. Run with "
        "--inspect, look at the printed module tree for the transformer "
        "block list, and add the correct dotted path to `candidates` above."
    )


def get_final_norm_and_head(model):
    """Same idea, for the final layernorm + LM head the logit lens needs."""
    norm_candidates = ["language_model.model.norm", "llm_backbone.llm.model.norm"]
    head_candidates = ["language_model.lm_head", "llm_backbone.llm.lm_head"]
    norm = head = None
    for path in norm_candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            norm = obj
            break
        except AttributeError:
            continue
    for path in head_candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            head = obj
            break
        except AttributeError:
            continue
    if head is None:
        # Most HF CausalLM classes expose this regardless of internal naming.
        head = model.get_output_embeddings()
    if norm is None:
        print("[warn] could not find a final norm module -- logit lens will "
              "skip normalization, which may skew results. Check --inspect output.")
    return norm, head


# ---------------------------------------------------------------------------
# Hidden-state capture via forward hooks
# ---------------------------------------------------------------------------

class HiddenStateRecorder:
    """
    Registers a forward hook on every decoder layer. Each hook call appends
    that layer's LAST-TOKEN hidden state (shape [d_model]) to a per-layer
    buffer. Call `.reset()` before each `predict_action(...)` call and
    `.stack()` after, to get a [n_gen_steps, n_layers, d_model] tensor
    covering that one env timestep.

    Only the last token position is kept, not the full sequence. During
    autoregressive generation with a KV cache, every step after the first is
    already seq_len=1; the first (prefill) pass processes the full
    prompt+image token sequence, and truncating that to the last position
    keeps exactly the hidden state that produced the first action token's
    logits -- the standard choice for logit-lens analysis, and the only
    thing keeping memory use bounded across a ~200-step episode.
    """

    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self._buffers = [[] for _ in range(self.n_layers)]
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            last_token = hs[:, -1, :].detach().to(torch.float32).cpu()
            self._buffers[layer_idx].append(last_token.squeeze(0))  # [d_model]
        return hook

    def reset(self):
        self._buffers = [[] for _ in range(self.n_layers)]

    def stack(self):
        """Returns [n_gen_steps, n_layers, d_model]. Raises if layers fired
        a different number of times than each other -- if that happens, the
        seq_len=1-per-step generation assumption above doesn't hold for this
        checkpoint and needs manual inspection before trusting saved data."""
        lengths = {len(b) for b in self._buffers}
        if len(lengths) != 1:
            raise RuntimeError(
                f"Layers recorded different step counts: {[len(b) for b in self._buffers]}."
            )
        per_layer = [torch.stack(b, dim=0) for b in self._buffers]  # [n_gen_steps, d_model]
        return torch.stack(per_layer, dim=1)  # [n_gen_steps, n_layers, d_model]

    def remove(self):
        for h in self._handles:
            h.remove()


def apply_logit_lens(hidden_vec, norm, head, topk=5):
    """
    Standard logit lens: take one layer's hidden state, run it through the
    model's OWN final norm + LM head (skipping every layer after the one
    you captured), and read off what it would have predicted.

    hidden_vec: [d_model] tensor.
    Returns (top_token_ids: list[int], top_probs: list[float]).
    """
    with torch.no_grad():
        target_dtype = next(head.parameters()).dtype
        target_device = next(head.parameters()).device
        x = hidden_vec.to(target_dtype).to(target_device)
        if norm is not None:
            x = norm(x)
        logits = head(x)
        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.topk(topk)
    return top_ids.tolist(), top_probs.tolist()


# ---------------------------------------------------------------------------
# LIBERO helpers (unchanged from the original smoke test)
# ---------------------------------------------------------------------------

def get_libero_env(task, resolution=256):
    env_args = {
        "bddl_file_name": task.bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def get_libero_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, :, :]  # LIBERO returns images flipped vertically
    return Image.fromarray(img)


def build_prompt(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--model_id", type=str,
                         default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--out_dir", type=str,
                         default="/fs/ess/PAS2324/alinaliu.12278/libero_smoketest")
    parser.add_argument("--inspect", action="store_true",
                         help="load the model, print its module tree, and exit. "
                              "Run this FIRST to verify find_decoder_layers() / "
                              "get_final_norm_and_head() match this checkpoint.")
    parser.add_argument("--save_every", type=int, default=1,
                         help="keep hidden states only every N env timesteps "
                              "(1 = every step). Raise this if a full episode "
                              "runs you out of memory or disk.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    print(f"[info] loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if args.inspect:
        print(model)
        return

    decoder_layers = find_decoder_layers(model)
    norm, head = get_final_norm_and_head(model)
    recorder = HiddenStateRecorder(decoder_layers)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(args.task_id)
    print(f"[info] task: {task_description!r}")

    env = get_libero_env(task, resolution=256)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    frames = []
    log = {"actions": [], "task": task_description, "success": False}
    episode_hidden_states = {}  # {acting_step_idx: tensor [n_gen_steps, n_layers, d_model]}

    t = 0
    step_idx = 0  # counts only "acting" steps (after the settle-in wait)
    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            t += 1
            continue

        img = get_libero_image(obs)
        frames.append(np.array(img))

        prompt = build_prompt(task_description)
        inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)

        recorder.reset()
        with torch.no_grad():
            action = model.predict_action(
                **inputs, unnorm_key=args.task_suite, do_sample=False
            )

        if step_idx % args.save_every == 0:
            try:
                episode_hidden_states[step_idx] = recorder.stack()
            except RuntimeError as e:
                print(f"[warn] step {step_idx}: {e}")

        log["actions"].append(np.asarray(action).tolist())

        obs, reward, done, info = env.step(action.tolist())
        t += 1
        step_idx += 1

        if done:
            log["success"] = True
            break

    recorder.remove()
    env.close()

    video_path = os.path.join(args.out_dir, f"{args.task_suite}_task{args.task_id}.mp4")
    imageio.mimsave(video_path, frames, fps=20)

    log_path = os.path.join(args.out_dir, f"{args.task_suite}_task{args.task_id}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    hidden_path = os.path.join(args.out_dir, f"{args.task_suite}_task{args.task_id}_hidden.pt")
    torch.save(episode_hidden_states, hidden_path)

    print(f"[done] success={log['success']}  steps={len(log['actions'])}")
    print(f"[done] video   -> {video_path}")
    print(f"[done] log     -> {log_path}")
    print(f"[done] hidden  -> {hidden_path}  "
          f"({len(episode_hidden_states)} timesteps saved)")


if __name__ == "__main__":
    main()