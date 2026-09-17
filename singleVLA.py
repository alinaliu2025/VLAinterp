"""
singleVLA.py

One OpenVLA + LIBERO episode, instrumented for interpretability.

Three modes, in the order you should run them:

  1. python singleVLA.py --inspect
        Loads the model, prints its module tree, exits. Does NOT import LIBERO.
        Run this first to confirm the attribute paths in find_decoder_layers()
        and get_final_norm_and_head() match this checkpoint.

  2. python singleVLA.py --static
        One predict_action on a flat grey frame with the hooks live, plus a
        logit-lens readout. Proves the whole capture path works. Still does
        NOT import LIBERO, so you can run it the moment the model loads.

  3. python singleVLA.py --task_suite libero_spatial --task_id 0
        The real episode: video, action log, hidden states. The only mode
        that touches the simulator.

What gets captured
------------------
A forward hook on each selected decoder layer records the LAST-TOKEN hidden
state on every forward pass. Inside one predict_action that is 1 prefill pass
over the prompt+image tokens, then one pass per generated action token (7 for
a 7-DoF action). Each env timestep therefore yields

    [n_gen_steps, n_layers, d_model]     with n_gen_steps == 8

Only the last token position is kept. During generation with a KV cache every
pass after the prefill is already seq_len=1, and truncating the prefill to its
final position keeps exactly the state that produced the first action token's
logits. That is the standard logit-lens choice and the only thing that keeps
memory bounded across a 200-step episode.

Storage
-------
All 32 layers at float16 is roughly 2MB per timestep, so a 220-step episode is
about 450MB. --layer_stride 4 cuts that to ~110MB. --save_every trades temporal
resolution for the same saving.
"""

import argparse
import json
import os

# Must be set before anything imports mujoco/robosuite. Compute nodes have no
# display, so rendering has to go through EGL.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# The finetuned LIBERO checkpoints ship no processor config of their own, so
# the processor always comes from the base repo. Loading it from the finetuned
# repo fails with "Unrecognized processing class".
BASE_REPO = "openvla/openvla-7b"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def pick_dtype(device):
    """bfloat16 on Ampere and newer, float16 on Volta, float32 on CPU."""
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(model_id, processor_id, device):
    dtype = pick_dtype(device)
    if device == "cuda":
        print(f"[info] gpu = {torch.cuda.get_device_name(0)}")
    print(f"[info] device = {device}, dtype = {dtype}")

    print(f"[info] processor <- {processor_id}")
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

    print(f"[info] weights   <- {model_id}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        # flash_attention_2 needs Ampere+ AND a compiled flash-attn. sdpa is
        # supported by this architecture and needs neither.
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model, dtype


# ---------------------------------------------------------------------------
# Model introspection. Verify with --inspect before trusting saved states.
# ---------------------------------------------------------------------------

def find_decoder_layers(model):
    """
    Locate the list of transformer decoder layers. Tries the attribute paths
    seen across OpenVLA/Prismatic forks and raises loudly if none match,
    rather than silently hooking nothing.
    """
    candidates = [
        "language_model.model.layers",      # typical HF causal-LM nesting
        "llm_backbone.llm.model.layers",    # some Prismatic-vlms trees
        "model.language_model.model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                print(f"[info] decoder layers at model.{path} ({len(obj)} layers)")
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not find the decoder layer list. Run with --inspect, find the "
        "transformer block list in the printed module tree, and add its dotted "
        "path to `candidates` above."
    )


def get_final_norm_and_head(model):
    """Same idea, for the final layernorm and LM head the logit lens needs."""
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
        print("[warn] no final norm found; logit lens will skip normalization, "
              "which may skew results. Check the --inspect output.")
    return norm, head


# ---------------------------------------------------------------------------
# Hidden-state capture
# ---------------------------------------------------------------------------

class HiddenStateRecorder:
    """
    Hooks the given layers. Call .reset() before each predict_action and
    .stack() after, to get [n_gen_steps, n_layers, d_model] for that timestep.

    store_dtype is float16 by default: these are activations for analysis, not
    values you compute with, and float32 doubles both RAM and disk for nothing.
    """

    def __init__(self, layers, layer_indices=None, store_dtype=torch.float16):
        self.layers = list(layers)
        self.n_layers = len(self.layers)
        self.layer_indices = (list(range(self.n_layers))
                              if layer_indices is None else list(layer_indices))
        self.store_dtype = store_dtype
        self._buffers = [[] for _ in range(self.n_layers)]
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(self.layers)
        ]

    def _make_hook(self, slot):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            last = hs[:, -1, :].detach().to(self.store_dtype).cpu()
            self._buffers[slot].append(last.squeeze(0))     # [d_model]
        return hook

    def reset(self):
        self._buffers = [[] for _ in range(self.n_layers)]

    def stack(self):
        """
        [n_gen_steps, n_layers, d_model].

        Raises if the layers fired different numbers of times, which would mean
        the one-pass-per-generated-token assumption does not hold for this
        checkpoint and the saved data cannot be trusted.
        """
        lengths = {len(b) for b in self._buffers}
        if len(lengths) != 1:
            raise RuntimeError(
                f"Layers recorded different step counts: "
                f"{[len(b) for b in self._buffers]}."
            )
        per_layer = [torch.stack(b, dim=0) for b in self._buffers]
        return torch.stack(per_layer, dim=1)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def apply_logit_lens(hidden_vec, norm, head, topk=5):
    """
    Take one layer's hidden state, run it through the model's OWN final norm
    and LM head, skipping every layer after the one captured, and read off what
    that layer would have predicted on its own.

    hidden_vec: [d_model]. Returns (top_token_ids, top_probs).
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


def build_prompt(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


def make_recorder(model, layer_stride):
    all_layers = find_decoder_layers(model)
    idx = list(range(0, len(all_layers), layer_stride))
    print(f"[info] hooking {len(idx)} of {len(all_layers)} layers: {idx}")
    return HiddenStateRecorder([all_layers[i] for i in idx], layer_indices=idx), idx


# ---------------------------------------------------------------------------
# Mode 2: static probe, no simulator
# ---------------------------------------------------------------------------

def run_static(args, processor, model, dtype, device):
    recorder, idx = make_recorder(model, args.layer_stride)
    norm, head = get_final_norm_and_head(model)

    img = Image.new("RGB", (256, 256), (127, 127, 127))
    inputs = processor(build_prompt(args.static_instruction), img).to(device, dtype=dtype)

    recorder.reset()
    with torch.no_grad():
        action = model.predict_action(
            **inputs, unnorm_key=args.unnorm_key, do_sample=False
        )
    hidden = recorder.stack()
    recorder.remove()

    print(f"[ok] action  = {np.asarray(action)}")
    print(f"[ok] shape   = {tuple(np.asarray(action).shape)}  (expect (7,))")
    print(f"[ok] hidden  = {tuple(hidden.shape)}  (gen_steps, layers, d_model)")
    print(f"[ok] layer_indices = {idx}")

    # Logit lens on the final generation step, sweeping the hooked layers.
    tok = getattr(processor, "tokenizer", None)
    print("\n[logit lens] final generation step, top-3 per hooked layer")
    for slot, layer_i in enumerate(idx):
        ids, ps = apply_logit_lens(hidden[-1, slot], norm, head, topk=3)
        if tok is not None:
            shown = [f"{tok.convert_ids_to_tokens(i)!r}({p:.2f})" for i, p in zip(ids, ps)]
        else:
            shown = [f"{i}({p:.2f})" for i, p in zip(ids, ps)]
        print(f"  layer {layer_i:2d}: " + "  ".join(shown))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "static_probe.pt")
    torch.save({"action": np.asarray(action), "hidden": hidden, "layer_indices": idx}, path)
    print(f"\n[done] wrote {path}")


# ---------------------------------------------------------------------------
# Mode 3: the real episode
# ---------------------------------------------------------------------------

def get_libero_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    # task.bddl_file is only a filename; the file lives under the suite's folder.
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def get_libero_image(obs):
    # Rotate 180 degrees, not just a vertical flip: this is what OpenVLA's
    # LIBERO eval does to match the training preprocessing.
    img = obs["agentview_image"][::-1, ::-1]
    return Image.fromarray(np.ascontiguousarray(img))


def to_libero_action(action):
    """
    OpenVLA predicts the gripper in [0, 1] with 1 = open. LIBERO wants
    -1 = open, +1 = close. Same two steps as OpenVLA's run_libero_eval.py:
    rescale to [-1, +1] and binarize, then flip the sign.
    """
    a = np.array(action, dtype=np.float64)
    a[-1] = np.sign(2.0 * a[-1] - 1.0)
    a[-1] *= -1.0
    return a


def run_rollout(args, processor, model, dtype, device):
    # Imported here, not at module top, so --inspect and --static work with no
    # simulator installed at all.
    import imageio
    from libero.libero import benchmark

    recorder, idx = make_recorder(model, args.layer_stride)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(args.task_id)
    print(f"[info] task: {task_description!r}")

    env = get_libero_env(task, resolution=args.resolution)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    frames = []
    # "actions" is the raw model output; "env_actions" is what was executed.
    log = {"actions": [], "env_actions": [], "task": task_description,
           "success": False}
    episode_hidden = {}

    t = 0
    step_idx = 0
    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            # Let the scene settle before the policy sees anything.
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            t += 1
            continue

        img = get_libero_image(obs)
        frames.append(np.array(img))

        inputs = processor(build_prompt(task_description), img).to(device, dtype=dtype)

        recorder.reset()
        with torch.no_grad():
            action = model.predict_action(
                **inputs, unnorm_key=args.unnorm_key, do_sample=False
            )

        if step_idx % args.save_every == 0:
            try:
                episode_hidden[step_idx] = recorder.stack()
            except RuntimeError as e:
                print(f"[warn] step {step_idx}: {e}")

        env_action = to_libero_action(action)
        log["actions"].append(np.asarray(action).tolist())
        log["env_actions"].append(env_action.tolist())
        obs, reward, done, info = env.step(env_action.tolist())
        t += 1
        step_idx += 1

        if step_idx % 25 == 0:
            print(f"[info] step {step_idx}")

        if done:
            log["success"] = True
            break

    recorder.remove()
    env.close()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.task_suite}_task{args.task_id}"

    video_path = os.path.join(args.out_dir, f"{stem}.mp4")
    imageio.mimsave(video_path, frames, fps=20)

    log_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    hidden_path = os.path.join(args.out_dir, f"{stem}_hidden.pt")
    torch.save({"hidden": episode_hidden, "layer_indices": idx}, hidden_path)

    print(f"[done] success={log['success']}  steps={len(log['actions'])}")
    print(f"[done] video  -> {video_path}")
    print(f"[done] log    -> {log_path}")
    print(f"[done] hidden -> {hidden_path}  ({len(episode_hidden)} timesteps)")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_suite", default="libero_spatial")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--model_id",
                   default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--processor_id", default=BASE_REPO,
                   help="processor always comes from the base repo; the "
                        "finetuned checkpoints ship no processor config")
    p.add_argument("--unnorm_key", default="libero_spatial",
                   help="key into the checkpoint's norm_stats")
    p.add_argument("--max_steps", type=int, default=220)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--layer_stride", type=int, default=1,
                   help="hook every Nth decoder layer. 1 = all 32.")
    p.add_argument("--save_every", type=int, default=1,
                   help="keep hidden states every N env timesteps")
    p.add_argument("--out_dir",
                   default="/fs/ess/PAS2324/alinaliu.12278/libero_smoketest")
    p.add_argument("--inspect", action="store_true",
                   help="print the module tree and exit; no LIBERO needed")
    p.add_argument("--static", action="store_true",
                   help="one predict_action on a grey frame; no LIBERO needed")
    p.add_argument("--static_instruction",
                   default="pick up the black bowl between the plate and the "
                           "ramekin and place it on the plate")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, dtype = load_model(args.model_id, args.processor_id, device)

    if args.inspect:
        print(model)
        return
    if args.static:
        run_static(args, processor, model, dtype, device)
        return
    run_rollout(args, processor, model, dtype, device)


if __name__ == "__main__":
    main()
