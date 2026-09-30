# VLA Project

## Week Sep 15-19
- Personal Thoughts: How should I be using AI specifically: using AI vs learning
    - Sometimes in getting faster results it backlashes and its harder to explain your results
        - Example from this week: an AI-suggested check said LIBERO was installed correctly, but it only passed because of the folder I ran it from. The real problem showed up later, inside a GPU job.
        - Example the other way: three bugs in the rollout code were found only by comparing line by line against OpenVLA's own evaluation code. Checking against the source is what made the result explainable.
    - Is it ok if I use claude code for OSC runs?
        - How it is set up right now: I log in to OSC myself, and Claude Code reuses that open connection. It never sees my password, and access ends when I close the window.
        - Rules it follows: it only touches my own folders, it explains each command, and I approve every command. It never runs anything that uses compute credits (`salloc`, `sbatch`, `srun`); it gives me those commands to run myself.
        - Question for Dr. Zhu: is this acceptable for the lab and under OSC's usage policy?
- Explain the current pipeline
    - Goal: run OpenVLA inside the LIBERO robot simulator and record what happens inside the model at every step, so we can ask whether early layers handle "perception" and later layers handle "action planning."
    - See the pipeline section below.

---

## What has been implemented so far

### The pipeline

```
 camera image (256x256)  +  instruction ("pick up the black bowl ...")
                 |
                 v
   OpenVLA-7B (finetuned on libero_spatial)
     vision encoders (DINOv2 + SigLIP) -> projector -> Llama-2 7B (32 layers)
                 |                                         |
                 |                          forward hooks on decoder layers
                 |                          save last-token hidden state (4096-d)
                 v                                         |
   7 action tokens -> de-normalized 7-DoF action           v
   (dx, dy, dz, droll, dpitch, dyaw, gripper)     hidden states per env step:
                 |                                [7 gen steps, n layers, 4096]
                 v
   LIBERO simulator (MuJoCo) executes the action, renders the next image
                 |
                 v
   outputs: video (.mp4), action log (.json), hidden states (.pt)
                 |
                 v
   analysis on laptop: convert .pt -> .npz, then peek.py
   (per-layer norms, cosine similarity to last layer, heatmap)
```

- **Model:** `openvla/openvla-7b-finetuned-libero-spatial`. The processor has to come from the base repo `openvla/openvla-7b`, because the finetuned repo doesn't ship one. Attention uses PyTorch SDPA, with no flash-attn.
- **Capture:** a forward hook on each chosen decoder layer saves the hidden state at the last token position on every forward pass. One `predict_action` call makes 7 forward passes, one per action token. So each environment step produces an array of shape `[7, n_layers, 4096]`: row k is the pass that produced action dimension k.
- **Logit lens:** in `--static` mode, each saved layer's hidden state is passed through the model's own final norm and output head, to see what that layer alone would predict.

### `singleVLA.py` has three modes

| Mode | What it does | Status |
|---|---|---|
| `--inspect` | Loads the model and prints its layer structure, to confirm where the hooks attach | Working |
| `--static` | One prediction on a flat grey image, with hooks and logit lens | **Working on OSC** (V100, fp16). Output shape `(7, 8, 4096)` with every 4th layer hooked |
| rollout | A full LIBERO episode: video, action log, hidden states | **Fixed this week, not run yet** |

### Environments

- **OSC (Pitzer), conda env `vla310`:** pinned to the versions the OpenVLA code expects (torch 2.2.2, transformers 4.40.1, timm 0.9.10, numpy 1.26.4).
- **Added this week, the LIBERO simulator:** LIBERO (commit `8f1084e`), mujoco 2.3.7, robosuite 1.4.1, plus matplotlib and h5py, which LIBERO needs but doesn't list.
    - Installed so no existing package version changed. This was checked by comparing package lists before and after each install.
    - `env.sh` now sets LIBERO's config path and `PYTHONPATH` automatically.
    - **Rendering is verified on a GPU node:** libero_spatial task 0 loads and renders correctly.
- **Laptop (Mac), conda env `vla310-local`:** same version pins without CUDA. Used for analysis (`peek.py`) and editing code, not for running the 7B model.

### Bugs fixed in the rollout code this week
All three were checked against OpenVLA's `run_libero_eval.py` and `libero_utils.py`.
1. **Task file path:** LIBERO only gives the file name. The full path is now built the way OpenVLA builds it. Before, the rollout would have crashed immediately.
2. **Image orientation:** the camera image is now rotated 180°, matching training, instead of only flipped vertically.
3. **Gripper convention:** the model outputs the gripper in [0, 1] with 1 = open, but LIBERO expects -1 = open and +1 = close. The value is now rescaled and sign-flipped before being sent to the robot. Before, the robot would have closed when the model meant open. The log saves both the raw model action and the executed action.

### A first look at the data (sanity check only)
From the `--static` run on a grey image, hooking layers 0, 4, ..., 28:
- The size of the internal representation (L2 norm) grows steadily with depth, from 2.5–4.2 at layer 0 to 107–124 at layer 28.
- Similarity to the last hooked layer stays low (0.13–0.46) through layer 16, then rises quickly: 0.54–0.63 at layer 20 and 0.72–0.81 at layer 24.
- This is a single blank image, so it only shows that the capture pipeline works. It is not a finding about perception vs. action.

---

## Open questions for Dr. Zhu
- **GPU precision:** Pitzer only has V100s, which can't run bfloat16, so the model runs in float16. The 32 GB V100s fit the model, which avoids moving to Cardinal or Ascend. Is the fp16/bf16 difference acceptable for interpretability results, or should final runs happen on A100/H100?
- **Center crop:** OpenVLA crops the center of each image before inference, because this checkpoint was trained with random crops. `singleVLA.py` doesn't crop yet. Should I match OpenVLA's setup exactly before collecting data?
- **Direction:** my original plan (below) was written before your synthesis. Which parts should change?
- **AI use:** is using Claude Code for OSC work, as described above, okay?

---

## Where I would like to go with this project
*(draft, to discuss)*

**Near term: get clean data**
1. Run one full rollout (libero_spatial task 0). Check the video, whether the task succeeds, and the saved hidden states.
2. Match OpenVLA's evaluation setup (center crop) and confirm the success rate looks reasonable over a few initial states.
3. Scale up to all 10 libero_spatial tasks with several initial states each, saving a subset of layers to keep storage manageable.

**Main question: is there a layer-wise split between perception and action planning?**
- **Logit lens over an episode:** at which layer does the final action token first become the top prediction? Does that layer change over the course of a task, for example approaching vs. grasping?
- **Linear probes per layer:**
    - *Perception targets:* object positions and gripper state, which LIBERO gives exactly from the simulator.
    - *Action targets:* the next action dimensions.
    - If perception information is decodable early and action information only late, that supports a split.
- **Compare across tasks and instructions:** does the same layer structure hold when the scene or wording changes?

**Longer term (depends on the discussion)**
- Compare with other VLA designs, or with World Action Models, which start from a video model instead of a language model (see `VLA_and_WAM_primer.md`).
- Connect to recent mechanistic work on VLAs, e.g. *Not All Features Are Created Equal* (arXiv 2603.19233).
