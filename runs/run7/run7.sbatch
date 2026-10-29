#!/bin/bash -l
# run7: where does the model think the gripper, the target bowl and the plate
# are, in the episodes that failed? Ridge probes trained on run2's SUCCESSFUL
# episodes read those three 3-D positions out of every layer's hidden state at
# the first-action-token position, then are read on the 8 failures that run6
# labelled (closed_away, missed_grasp, ...). Headline folds leave a whole task
# out, so a probe cannot answer from "which task is this" alone. CPU only, no
# rollouts: reads runs/run2/, writes only runs/run7/.
#
#   1. --mode failprobe: fit + predict, all 33 layers, two fold schemes
#      (leave-one-task-out, and by episode) -> runs/run7/failprobe/results.npz
#   2. --mode failprobe --analyze: plots + CSVs in runs/run7/plots/
#      (failprobe_close.png is the main figure)
#
# Resumable: step 1 is skipped once its results exist.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run7/run7.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run7/run7_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
git log --oneline -1
sha256sum experiments.py
date
status=0
if [ ! -f runs/run7/failprobe/results.npz ]; then
    python experiments.py --mode failprobe --src_dir runs/run2 --out_dir runs/run7
    status=$?
fi
date
if [ ${status} -eq 0 ]; then
    python experiments.py --mode failprobe --analyze --out_dir runs/run7
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
