#!/bin/bash
# One-time (re-run anytime to pick up a requirements.txt change) setup of
# the isolated venv used by sft_stage{1,2,3}.sh. Run this yourself, directly
# -- do NOT submit it with sbatch: it needs to reach PyPI, and on most SLURM
# clusters only login/data-transfer nodes have internet access, not compute
# nodes.
#
#   bash script_examples/setup_env.sh
#
# See cluster_env.sh for what this venv is and why it's built this way.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

echo "Container:  ${CONTAINER}"
echo "Venv:       ${VENV_DIR}"

if [ ! -x "${VENV_PY}" ]; then
  echo "Creating venv..."
  apptainer exec ${APPTAINER_BINDS} "${CONTAINER}" python3 -m venv --system-site-packages "${VENV_DIR}"
fi

apptainer exec ${APPTAINER_BINDS} "${CONTAINER}" "${VENV_PY}" -m pip install --upgrade pip
apptainer exec ${APPTAINER_BINDS} "${CONTAINER}" "${VENV_PY}" -m pip install -r "${REPO_DIR}/requirements.txt"

echo
echo "Done. torch/torchvision are inherited from the container (requirements.txt"
echo "leaves them unpinned, so pip left the container's build alone) -- verify with:"
apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" "${VENV_PY}" -c \
  "import torch; print('torch', torch.__version__, 'from', torch.__file__); print('cuda available:', torch.cuda.is_available())"
