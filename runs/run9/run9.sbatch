#!/bin/bash -l
# run9: run6's last-position instruction steering, on the GENERIC instruction set.
# Adds mean_B - mean_A (run5 generic cluster means, fit on inits 1-4) at the output
# of one decoder layer, last prompt position, prefill only; tests on the 10 held-out
# init-0 images. 10 x 9 instruction pairs x 10 images at layers 7,11,15,31 (the run6
# layers that carried the result), plus a same-norm random-direction control.
# Needs runs/run5/grid/generic.npz. Model: libero-spatial fine-tune (as run5/run6).
#
# Runs from JACK'S clone. Reads Alina's env/weights, writes only here.
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp && sbatch runs/run9/run9.sbatch
# Redo the analysis later (CPU only):
#   python steer.py --analyze --out_dir runs/run9
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run9
#SBATCH --partition=gpudebug-exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run9/run9_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py steer.py runs/run5/grid/generic.npz
date
python steer.py --grid runs/run5/grid/generic.npz --layers 7,11,15,31 --out_dir runs/run9
status=$?
date
if [ ${status} -eq 0 ]; then
    python steer.py --analyze --out_dir runs/run9
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
