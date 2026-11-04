#!/bin/bash -l
# run10: linear probes (CPU only) on the saved last-prompt-position hidden states
# of run5 (libero-spatial), run7 (libero-goal) and run8 (base), generic and suite sets.
# At every readout: a ridge readout 4096 -> 10 for the scene and for the instruction,
# scored on held-out rows. v1 new placements (train inits 1-4, test init 0);
# v2 new partner (scene probe tested on unseen instructions, instruction probe on
# unseen scenes, 5 folds of 2; tested on init 0 only, never seen in training).
# Control: shuffled training labels (3x).
#
# Runs from JACK'S clone. Reads the run5/7/8 grids, writes only runs/run10/.
#   cd /fs/ess/PAS2324/wiencek.16/VLAinterp && sbatch runs/run10/run10.sbatch
#
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_run10
#SBATCH --partition=cpu-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o runs/run10/run10_%j.log
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from = your clone
source ./my_env.sh
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
git log --oneline -1
sha256sum tdnv_present.py experiments.py tdnv_iv_analyze.py
date
status=0
for spec in "run5|OpenVLA libero-spatial fine-tune" "run7|OpenVLA libero-goal fine-tune" "run8|OpenVLA base (no LIBERO fine-tune)"; do
    run=${spec%%|*}; label=${spec#*|}
    for set in generic suite; do
        echo "=== ${run} ${set}"
        python tdnv_present.py probe --grid runs/${run}/grid/${set}.npz --out_dir runs/run10/${run} --label "${label}" || status=$?
        date
    done
done
echo "[job] python exit code ${status}"
exit ${status}
