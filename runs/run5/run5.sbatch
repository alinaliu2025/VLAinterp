#!/bin/bash -l
# run5: TDNV image vs. instruction (spec: tdnv_image_vs_instruction.md).
# Crossed grid: 10 libero_spatial layouts x 5 init states = 50 images, times two
# instruction sets of 10 ("suite" = the 10 libero_spatial instructions, "generic" =
# 10 different tasks plausible in every layout). Hidden state at the last prompt
# position, all 33 readouts, float32. Config: runs/run5/config.json.
#
# Runs from JACK'S clone. Reads Alina's env/weights/LIBERO, writes only here.
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp && sbatch runs/run5/run5.sbatch
# Redo the analysis later (CPU only):
#   python tdnv_iv_analyze.py --config runs/run5/config.json --out_dir runs/run5
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run5/run5_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py tdnv_iv_extract.py tdnv_iv_analyze.py runs/run5/config.json
date
python tdnv_iv_extract.py --config runs/run5/config.json --out_dir runs/run5
status=$?
date
if [ ${status} -eq 0 ]; then
    python tdnv_iv_analyze.py --config runs/run5/config.json --out_dir runs/run5
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
