#!/bin/bash -l
# run2: run1 again with the logit-lens tie-break fix. --mode lens on all 50
# libero_spatial episodes (tasks 0-9 x inits 0-4): logit lens at every layer,
# hidden states for --mode probe, success rate in episodes.csv. Resumable:
# resubmitting this same script skips finished episodes, and it stays run2.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run2/run2.sbatch
#
# Changes from run1.sbatch: the job now exits with Python's exit code, so a
# crash shows up as FAILED in sacct instead of COMPLETED.
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run2
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run2/run2_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py
date
python experiments.py --mode lens --tasks 0-9 --inits 0-4 --save_video --out_dir runs/run2
status=$?
date
echo "[job] python exit code ${status}"
exit ${status}
