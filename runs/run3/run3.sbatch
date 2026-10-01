#!/bin/bash -l
# run3: --mode patchlens on task 0 x inits 0-4 (5 episodes). Everything run2
# records (action lens, hidden states, success), plus a logit lens on the 256
# image-patch positions over the text vocabulary at every step and layer, the
# 224x224 model input image per step, and raw patch hidden states every 20
# steps (for Prismatic's head or a tuned lens later). Resumable: resubmitting
# this same script skips finished episodes, and it stays run3.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run3/run3.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run3
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run3/run3_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py
date
python experiments.py --mode patchlens --tasks 0 --inits 0-4 --save_video --out_dir runs/run3
status=$?
date
echo "[job] python exit code ${status}"
exit ${status}
