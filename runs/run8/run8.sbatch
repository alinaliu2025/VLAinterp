#!/bin/bash -l
# run8: run5 (TDNV image vs. instruction) repeated with the BASE model (no LIBERO fine-tune).
# Same 50 libero_spatial first frames, same two instruction sets (suite, generic),
# same last-prompt-position hidden states and analysis. Only model_id / unnorm_key
# change (runs/run8/config.json). Weights come from Jack's own HF cache
# (/fs/ess/PAS2324/wiencek.16/hf_cache, downloaded like Alina's hf_cache).
#
# Runs from JACK'S clone. Reads Alina's env/LIBERO, writes only here.
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp && sbatch runs/run8/run8.sbatch
# Redo the analysis / presentation figures later (CPU only):
#   python tdnv_iv_analyze.py --config runs/run8/config.json --out_dir runs/run8
#   python tdnv_present.py curves --csv_dir runs/run8/plots --out_dir runs/run8/present --label "OpenVLA base (no LIBERO fine-tune)"
#   python tdnv_present.py scatter --grid runs/run8/grid/generic.npz --out_dir runs/run8/present --label "OpenVLA base (no LIBERO fine-tune)"
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run8/run8_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
export HF_HOME=/fs/ess/PAS2324/wiencek.16/hf_cache   # Jack's cache: base + libero-goal
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py tdnv_iv_extract.py tdnv_iv_analyze.py tdnv_present.py runs/run8/config.json
date
python tdnv_iv_extract.py --config runs/run8/config.json --out_dir runs/run8
status=$?
date
if [ ${status} -eq 0 ]; then
    python tdnv_iv_analyze.py --config runs/run8/config.json --out_dir runs/run8
    status=$?
fi
if [ ${status} -eq 0 ]; then
    # Same model inputs as run5? (exact pixel match of all 50 images)
    python -c "import numpy as np; a=np.load('runs/run5/grid/generic.npz'); b=np.load('runs/run8/grid/generic.npz'); \
print('[check] images identical to run5:', a['images'].shape == b['images'].shape and bool((a['images'] == b['images']).all()), \
'| image_src identical:', bool((a['image_src'] == b['image_src']).all()))"
    python tdnv_present.py curves --csv_dir runs/run8/plots --out_dir runs/run8/present --label "OpenVLA base (no LIBERO fine-tune)" && \
    python tdnv_present.py scatter --grid runs/run8/grid/generic.npz --out_dir runs/run8/present --label "OpenVLA base (no LIBERO fine-tune)"
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
