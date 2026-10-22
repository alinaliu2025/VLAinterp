#!/bin/bash -l
#SBATCH --account=PAS2324
#SBATCH --job-name=vla_dryrun
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH -o results/logs/dryrun_%j.log
cd /fs/ess/PAS2324/alinaliu.12278/VLAinterp
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "===== RUN A ====="; date
python experiments.py --run A --dry_run --save_video || echo RUN_A_FAILED
echo "===== RUN B ====="; date
python experiments.py --run B --dry_run --save_video || echo RUN_B_FAILED
echo "===== FILES ====="; date
find results/libero_spatial/dry_run -type f -exec ls -la {} +
cat results/libero_spatial/dry_run/episodes.csv
