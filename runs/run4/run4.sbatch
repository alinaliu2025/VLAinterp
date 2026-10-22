#!/bin/bash -l
# run4: tuned lens on run2's saved hidden states, to look for a gap between
# perception and action at the first-action-token position (the one that
# produces x). No rollouts: everything is read from runs/run2/hidden/ and
# runs/run2/lens/, and only runs/run4/ is written.
#
#   1. --mode tunedlens: for every layer 0-31, train an affine translator
#      A x + b in front of the frozen final RMSNorm + the 256 action rows of
#      lm_head, to match the model's own final x distribution (KL). Variants
#      bias / rank 8, 32, 128 / full, plus a shuffled-target control; 5 folds
#      by episode, scored on held-out episodes. Loads the model once only to
#      cache the read-out (runs/run4/tunedlens/readout.npz).
#   2. --mode probe: ridge probes for scene state (gripper, objects) and the
#      executed action on the same hidden states = the perception curve.
#   3. --mode tunedlens --analyze: replot, now with the gap plot
#      (plots/gap_perception_action.png, plots/gap_summary.csv).
#
# Resumable: steps whose output already exists are skipped, so resubmitting
# this same script after a timeout finishes the job, and it stays run4.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run4/run4.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run4/run4_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py
date
status=0
if [ ! -f runs/run4/tunedlens/results.npz ]; then
    python experiments.py --mode tunedlens --src_dir runs/run2 --out_dir runs/run4
    status=$?
fi
date
if [ ${status} -eq 0 ] && [ ! -f runs/run4/plots/probe_r2.csv ]; then
    python experiments.py --mode probe --src_dir runs/run2 --out_dir runs/run4
    status=$?
fi
date
if [ ${status} -eq 0 ]; then
    python experiments.py --mode tunedlens --analyze --out_dir runs/run4
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
