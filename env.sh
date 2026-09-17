# source this at the top of every session, on login nodes and compute nodes
#
# Pitzer requires an explicit miniconda3 version. If this one disappears,
# find the current one with:  module spider miniconda3
module reset
module load miniconda3/25.11.1-py312

# `conda activate` on a -p path needs the shell hook; `source activate` is the
# older API and breaks in non-interactive shells and batch jobs.
eval "$(conda shell.bash hook)"
conda activate /fs/ess/PAS2324/alinaliu.12278/conda/envs/vla310

export CONDA_PKGS_DIRS=/fs/ess/PAS2324/alinaliu.12278/.conda_pkgs
export HF_HOME=/fs/ess/PAS2324/alinaliu.12278/hf_cache
export PYTHONNOUSERSITE=1

# headless rendering, needed once LIBERO is in the picture
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# LIBERO. The config file stops it prompting for a dataset path on first
# import. PYTHONPATH is needed because `pip install -e` of LIBERO does not
# make `import libero` work from outside the repo directory.
export LIBERO_CONFIG_PATH=/fs/ess/PAS2324/alinaliu.12278/.libero
export PYTHONPATH=/fs/ess/PAS2324/alinaliu.12278/LIBERO${PYTHONPATH:+:$PYTHONPATH}

echo "[env] python: $(which python)"
