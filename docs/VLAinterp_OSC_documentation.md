# VLAinterp on OSC — What's Actually Going On

## 1. The research goal:

Extending an interpretability method (logit lens / linear probing, the same family as [[dprobe]] work on LLM deception probes) to VLA models.  
QUESTION: inside a VLA, is there a layer-wise separation between "perception" (understanding the scene) and "action-planning" (deciding what to do)? (FYI: I made this before Dr. Zhu's idea/synthesis so everything is changable)  
The concrete first step is OpenVLA (7B), a model that takes an image + text instruction and outputs a 7-DoF robot action, evaluated against the LIBERO simulation benchmark. 

## 2. How peices connect:

```
                     ┌───────────────────────────────────────────────┐
                     │   OSC project space (PAS2324)                 │
                     │   /fs/ess/PAS2324/alinaliu.12278/             │
                     │                                               │
                     │   conda/envs/vla310/  <- the env              │
                     │   hf_cache/           <- model weights        │ (hf_cache so no reinstallation necessary)
                     │   VLAinterp/          <- your repo            │ 
                     │   libero_smoketest/   <- outputs dir          │
                     └───────────────┬───────────────────────────────┘
                                     │ 
                                     │ 
                     ┌───────────────┴────────────────────┐
                     │   compute node (via salloc)        │
                     │   source env.sh → activates vla310 │
                     │   runs singleVLA.py                │
                     └────────────────────────────────────┘
```

(FYI): From what I understand, project space is one shared filesystem mounted on all three OSC GPU clusters — Pitzer, Cardinal (H100s), and Ascend (A100s). So, env.sh and hf_cache don't need to be redone per cluster; you just `salloc` on whichever cluster has the GPU you need and everything is already there.

## 3. Directory map (components and what they do): 

`/fs/ess/PAS2324/alinaliu.12278/` contains:

| Directory | What it is |
|---|---|
| `VLAinterp/` | Git repo & actual code (explained in later section) |
| `conda/envs/vla310/` | Only conda env this project uses |
| `hf_cache/` | HuggingFace model cache (`HF_HOME` points here) |
| `libero_smoketest/` | Default output directory for `singleVLA.py` runs (videos, action logs, saved hidden states) |

Inside `VLAinterp/`:

```
VLAinterp/
├── docs/
│   └── settingUpVLA&LIBERO.md    ← personal setup notes
├── env.sh                        ← environment bootstrap script, source this every session
├── singleVLA.py                  ← the actual experiment script
└── static.log                    ← output from a past --static run
```

**The current repo is clean and up to date. (https://github.com/alinaliu2025/VLAinterp)** .

## 4. Environment setup (what `env.sh` does): 

File you `source` at the start of every session (login node or compute node):

```bash
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
```

What/Why:

```bash
module reset #grabbing one specific version of conda
module load miniconda3/25.11.1-py312 # clears off anything you'd previously borrowed
```
Pitzer requires loading a specific pinned miniconda3 module before `conda` exists on your PATH at all. 
```bash
eval "$(conda shell.bash hook)"
conda activate /fs/ess/PAS2324/alinaliu.12278/conda/envs/vla310
```

```bash
export CONDA_PKGS_DIRS=/fs/ess/PAS2324/alinaliu.12278/.conda_pkgs
export HF_HOME=/fs/ess/PAS2324/alinaliu.12278/hf_cache
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
- `CONDA_PKGS_DIRS` and `HF_HOME` redirect conda's package cache and HuggingFace's model cache into your project space instead of your home directory.
- `PYTHONNOUSERSITE=1` stops Python from picking up anything installed in `~/.local`
- `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` 
  - A simulator like MuJoCo works out where every object in the virtual scene is (the robot arm, the bowl, the table, etc.), and then it needs to turn that 3D information into a flat 2D picture ("rendering")

  - EGL is a way of doing that same math but instead, the finished picture (just a grid of pixel color values) gets written into a chunk of memory, like a photo sitting in RAM, never displayed anywhere.
  - That block of memory is then exactly what gets handed to OpenVLA as its "image" input.
- (FYI): Right now, `libero_smoketest/static_probe.pt` came from the `--static` mode, which doesn't touch LIBERO, robosuite, or MuJoCo at all — it never asks the simulator for anything. Instead, singleVLA.py just manufactures a **flat grey square directly in Python (Image.new("RGB", (256, 256), (127, 127, 127))**

## 5. What's actually installed in `vla310`:

Setup doc (`docs/settingUpVLA&LIBERO.md`) :

| package | pin | reason |
|---|---|---|
| transformers | 4.40.1 | OpenVLA's Hub-hosted `modeling_prismatic.py` targets this exact API |
| tokenizers | 0.19.1 | required range for that transformers version |
| timm | 0.9.10 | the DINOv2+SigLIP vision backbone uses the 0.9 API |
| numpy | <2 (confirmed: 1.26.4) | torch 2.2 is compiled against numpy 1.x |
| huggingface_hub | 0.23.5 | contemporaneous with transformers 4.40 |
| mujoco | 2.3.7 | robosuite 1.4.1-era; MuJoCo ≥3.4 silently changes LIBERO's initial states |
| robosuite | 1.4.1 | what LIBERO expects |
| torch | 2.2.2+cu121 (confirmed) | |
| torchvision | 0.17.2+cu121 (confirmed) | |



`vla310` has everything OpenVLA (the model itself) needs to run —
`transformers`, `tokenizers`, `timm`, `numpy`, `huggingface_hub`, `torch`,
`torchvision`.

What's **not yet installed** is the simulator side: `libero`, `robosuite`,
and `mujoco`.

## 6. Checkpoints and cache

`HF_HOME=/fs/ess/PAS2324/alinaliu.12278/hf_cache` :

| cached repo | size |
|---|---|
| `openvla/openvla-7b` | 2.4M (processor/config only, not full weights) |
| `openvla/openvla-7b-finetuned-libero-spatial` | 15G (the actual model weights you run) |

Two things worth knowing about this pairing:
- The finetuned checkpoint ships **no processor config of its own**. `AutoProcessor.from_pretrained` on the finetuned repo raises "Unrecognized processing class." So `singleVLA.py` always loads the processor from the base repo (`openvla/openvla-7b`) and the weights from the finetuned one, that's why both are cached, even though you only ever run the finetuned model.
- `unnorm_key="libero_spatial"` is required at inference time to un-normalize the model's raw action outputs back into real robot-action units. Basically, it's the only key in that checkpoint's `norm_stats`, so it's not really a "choice," just the one valid value.
## 7. `singleVLA.py` — what it does

Three modes:

```bash
python singleVLA.py --inspect
```
Loads the model, prints its full module tree, exits immediately. No LIBERO import at all. 

Purpose: confirm the attribute paths used to find the decoder layers (`find_decoder_layers()`) and the final norm/head (`get_final_norm_and_head()`) actually match this checkpoint's internal structure before trusting anything captured from it. 
```bash
python singleVLA.py --static
```
One `predict_action` call on a flat grey 256×256 image, with forward hooks live on 8 of the 32 decoder layers (every 4th, via `--layer_stride`, default 1 = all layers but the smoketest run used stride 4 based on `static.log`'s `[0, 4, 8, 12, 16, 20, 24, 28]`). Still no LIBERO import. 

Purpose: prove the entire hook → generate → capture → logit-lens pipeline works before spending a GPU allocation on the simulator.

**Confirmed working** — `static.log` in the repo shows a full successful run: model loaded (4 checkpoint shards, ~70s), 8 layers hooked, a valid 7-DoF action produced (`[0.089, 0.155, -0.186, -0.069, -0.006, -0.010, 0.996]`), hidden states shaped correctly `(7, 8, 4096)` = (generation steps, hooked layers, hidden dim), and results saved to `libero_smoketest/static_probe.pt`. 



The natural next thing to verify is that we'll need to `salloc` on Cardinal (H100) or Ascend (A100) rather than Pitzer if you land a 16GB V100, since that won't fit the 7B model

```bash
python singleVLA.py --task_suite libero_spatial --task_id 0
```
**NOT RUN YET:** runs a full LIBERO episode (up to `--max_steps`, default 220), rendering headless via EGL, recording an action at every timestep, saving hidden states every `--save_every` steps, and writing out a video (`.mp4`), an action/success log (`.json`), and hidden states (`.pt`) to `--out_dir` (defaults to `libero_smoketest/`). 


Storage note baked into the script's design: capturing all 32 layers at float16 costs ~2MB/timestep, so a full 220-step episode is ~450MB — manageable, but `--layer_stride 4` (as already used in the static test) or `--save_every 2` cuts that down further.

## 8. Getting a GPU

```bash
salloc --account=PAS2324 --nodes=1 --gpus-per-node=1 --mem=64G --time=02:00:00
nvidia-smi --query-gpu=name,memory.total --format=csv
```



## 9. Quick reference

```bash
# every session, every node
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh

# get a GPU (Pitzer may give you a V100 — too small; use cardinal.osc.edu or
# ascend.osc.edu directly if you need to guarantee H100/A100)
salloc --account=PAS2324 --nodes=1 --gpus-per-node=1 --mem=64G --time=02:00:00
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp

# in order of increasing scope
python singleVLA.py --inspect
python singleVLA.py --static
python singleVLA.py --task_suite libero_spatial --task_id 0
```


