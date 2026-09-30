#!/bin/bash -l
# run1: --mode lens on all 50 libero_spatial episodes (tasks 0-9 x inits 0-4).
# Logit lens at every layer, hidden states saved for --mode probe, and the
# success rate in episodes.csv. Resumable: resubmitting this same script skips
# finished episodes, and it stays run1.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run1/run1.sbatch
#
# GPU request, shell and env setup copied from run0.sbatch, which worked.
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run1/run1_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
sha256sum experiments.py singleVLA.py
date
python experiments.py --mode lens --tasks 0-9 --inits 0-4 --save_video --out_dir runs/run1
date
