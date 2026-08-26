# source this at the top of every session
module reset
module load miniconda3
export CONDA_PKGS_DIRS=/fs/ess/PAS2324/alinaliu.12278/.conda_pkgs
export HF_HOME=/fs/ess/PAS2324/alinaliu.12278/hf_cache
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
source activate /fs/ess/PAS2324/alinaliu.12278/conda/envs/vla310
