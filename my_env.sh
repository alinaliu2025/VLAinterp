# Jack's session setup. Reuses Alina's conda env, model weights and LIBERO
# (read-only) and keeps everything this session WRITES in Jack's own folder.
#   source ~/VLAinterp/my_env.sh      (or wherever you cloned the repo)

# 1. Alina's setup: conda env, HF_HOME, LIBERO paths, EGL. Only reads her files.
source /fs/ess/PAS2324/alinaliu.12278/VLAinterp/env.sh

# 2. Never touch the network or her cache: the weights are already downloaded.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3. trust_remote_code copies OpenVLA's model code into a modules cache.
#    Point that at Jack's own folder instead of Alina's hf_cache.
export HF_MODULES_CACHE="$HOME/.cache/hf_modules"
mkdir -p "$HF_MODULES_CACHE"

echo "[my_env] code + outputs: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
