# Setting up OpenVLA + LIBERO on OSC

Rebuilt from scratch 2026-08-26. The old `openvla` conda env and the old
instructions in this file are gone; do not resurrect them.

## Every session

```bash
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
```

which is:

```bash
module reset
module load miniconda3
export CONDA_PKGS_DIRS=/fs/ess/PAS2324/alinaliu.12278/.conda_pkgs
export HF_HOME=/fs/ess/PAS2324/alinaliu.12278/hf_cache
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
source activate /fs/ess/PAS2324/alinaliu.12278/conda/envs/vla310
```

## Getting a GPU

```bash
salloc --account=PAS2324 --nodes=1 --gpus-per-node=1 --mem=64G --time=02:00:00
nvidia-smi --query-gpu=name,memory.total --format=csv
```

A 16GB V100 will not fit the 7B model. If that is what you get, exit and
resubmit on `cardinal.osc.edu` (H100) or `ascend.osc.edu` (A100). The project
filesystem is shared across all three clusters, so nothing reinstalls.

## The pins, and why

| package | pin | reason |
|---|---|---|
| transformers | 4.40.1 | the Hub's `modeling_prismatic.py` targets this API |
| tokenizers | 0.19.1 | required range of that transformers |
| timm | 0.9.10 | the DINOv2+SigLIP backbone uses the 0.9 API |
| numpy | <2 | torch 2.2 is compiled against numpy 1.x |
| huggingface_hub | 0.23.5 | contemporaneous with transformers 4.40 |
| mujoco | 2.3.7 | robosuite 1.4.1 era; mujoco >=3.4 silently changes libero_spatial initial states |
| robosuite | 1.4.1 | what LIBERO expects |

**Never run `pip install -r requirements.txt` from inside the LIBERO repo.**
It pins `transformers==4.21.1` and `numpy==1.22.4` and will destroy this
environment. Install LIBERO with `--no-deps` and add its dependencies by hand.

## No flash-attn

Optional, needs Ampere or newer, takes half an hour to compile. The model
declares SDPA support. `singleVLA.py` uses `attn_implementation="sdpa"`.

## Checkpoint notes

- Weights: `openvla/openvla-7b-finetuned-libero-spatial`, already in `hf_cache`.
- The processor must be loaded from `openvla/openvla-7b`. The finetuned repos
  ship no processor config and `AutoProcessor` on them raises
  "Unrecognized processing class".
- `unnorm_key` for that checkpoint is `libero_spatial`. It is the only key in
  its `norm_stats`.
- The finetuned config's `auto_map` points at the base repo, so the first run
  needs Hub access for a few MB of `.py` files. Warm it on a login node.

## Running singleVLA.py

```bash
python singleVLA.py --inspect    # module tree, no LIBERO needed
python singleVLA.py --static     # one action + hidden states, no LIBERO needed
python singleVLA.py --task_suite libero_spatial --task_id 0
```

Storage: all 32 layers at float16 is ~2MB per timestep, so a 220-step episode
is ~450MB. Use `--layer_stride 4` or `--save_every 2` to cut that down.
