#!/bin/bash -l
# run4: instruction swap (Dr. Zhu, 2026-10). Same 50 images (first frame of
# libero_spatial task 0, inits 0-49), every libero_spatial instruction plus a
# paraphrase control and a changed-goal control. Hidden states at the
# first-action-token position, all layers.
#
# Runs from JACK'S clone. Reads Alina's env/weights/LIBERO, writes only here.
#   cd ~/VLAinterp && sbatch runs/run4/run4.sbatch
# The job also runs the analysis at the end. To redo it later (CPU only):
#   python instruction_swap.py --analyze --out_dir runs/run4 --task 0
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run4/run4_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py instruction_swap.py
date
python instruction_swap.py --task 0 --inits 0-49 --out_dir runs/run4 \
    --extra "grab the black bowl between the plate and the ramekin and put it on the plate" \
    --extra "pick up the black bowl between the plate and the ramekin and place it on the cookie box"
status=$?
date
# CPU analysis right away (a few minutes), only if collection succeeded.
if [ ${status} -eq 0 ]; then
    python instruction_swap.py --analyze --out_dir runs/run4 --task 0
    status=$?
fi
echo "[job] python exit code ${status}"
exit ${status}
