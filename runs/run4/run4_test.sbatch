#!/bin/bash -l
# run4 pipeline test: 2 images (inits 0-1), same instructions as run4, then the
# analysis. The numbers mean nothing at N=2; this only checks that collection
# and analysis run end to end before spending GPU time on the full run4.
#
# Queue the test, and the full run4 to start only if the test succeeds:
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp
#   jid=$(sbatch --parsable runs/run4/run4_test.sbatch)
#   sbatch --dependency=afterok:${jid} runs/run4/run4.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run4_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run4/run4_test_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1
sha256sum experiments.py singleVLA.py instruction_swap.py
date
python instruction_swap.py --task 0 --inits 0-1 --out_dir runs/run4_test \
    --extra "grab the black bowl between the plate and the ramekin and put it on the plate" \
    --extra "pick up the black bowl between the plate and the ramekin and place it on the cookie box"
status=$?
if [ ${status} -eq 0 ]; then
    python instruction_swap.py --analyze --out_dir runs/run4_test --task 0
    status=$?
fi
date
echo "[job] python exit code ${status}"
exit ${status}
