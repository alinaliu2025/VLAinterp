# Local (macOS) counterpart to env.sh. Source it from the repo root:
#
#   source env_local.sh
#
# First time only:  conda env create -f environment-local.yml

eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.zsh hook)"
conda activate vla310-local

# same reason as on OSC: ignore anything pip-installed into ~/.local
export PYTHONNOUSERSITE=1

echo "[env] python: $(which python)"
