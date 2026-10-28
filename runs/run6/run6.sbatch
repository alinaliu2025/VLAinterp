#!/bin/bash -l
# run6: instruction steering at the LAST prompt position only (follow-up to run5).
# Adds mean_B - mean_A (run5 suite cluster means, fit on inits 1-4) at the output
# of one decoder layer, last prompt position, prefill only; tests on the 10 held-out
# init-0 images. 10 x 9 instruction pairs x 10 images at layers 3,5,7,9,11,15,23,31,
# plus a same-norm random-direction control. Needs runs/run5/grid/suite.npz.
#
# Runs from JACK'S clone. Reads Alina's env/weights, writes only here.
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp && sbatch runs/run6/run6.sbatch
# Redo the analysis later (CPU only):
#   python steer.py --analyze --out_dir runs/run6
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run6
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run6/run6_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py steer.py runs/run5/grid/suite.npz
date
python steer.py --grid runs/run5/grid/suite.npz --out_dir runs/run6
status=$?
date
if [ ${status} -eq 0 ]; then
    python steer.py --analyze --out_dir runs/run6
    status=$?
fi
echo "[job] python exit code ${status}"
exit ${status}
