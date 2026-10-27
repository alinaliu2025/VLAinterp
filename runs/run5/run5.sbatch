#!/bin/bash -l
# run5: logit lens over the FULL vocabulary on run2's saved hidden states, at
# the first-action-token position (the one that produces x). Every earlier lens
# softmaxed over the 256 action rows only; this one asks how much probability
# goes to ordinary words vs action bins at each layer, and where bins take
# over. No rollouts: reads runs/run2/hidden/ and runs/run2/lens/, writes only
# runs/run5/.
#
#   1. --mode vocablens: loads the model once to cache the whole lm_head + final
#      norm + token strings (runs/run5/vocablens/readout.npz, ~270 MB), then for
#      every layer splits the full-vocab softmax into bins / text / pad / special
#      and keeps the top-1 token, top-5 text tokens and the bins-only argmax.
#   2. --mode vocablens --analyze: plots + CSVs in runs/run5/plots/
#      (vocablens_mass.png is the main figure; crossover layer in
#      vocablens_summary.csv).
#
# Resumable: step 1 is skipped once its results exist, so resubmitting this
# same script after a timeout finishes the job, and it stays run5.
#
#   cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp && sbatch runs/run5/run5.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /fs/ess/PAS2324/alinaliu.12278/VLAinterp/runs/run5/run5_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py
date
status=0
if [ ! -f runs/run5/vocablens/results.npz ]; then
    python experiments.py --mode vocablens --src_dir runs/run2 --out_dir runs/run5
    status=$?
fi
date
if [ ${status} -eq 0 ]; then
    python experiments.py --mode vocablens --analyze --out_dir runs/run5
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
