# Shared config for the SLURM/apptainer scripts in this directory. Sourced
# by setup_env.sh and sft_stage{1,2,3}.sh -- not meant to be run directly.
#
# Env model: the apptainer image supplies the underlying system (CUDA
# driver userspace, a matching torch/torchvision build, system libraries).
# On top of that we keep one dedicated, fully isolated virtualenv for this
# project's own dependencies (transformers, trl, deepspeed, ...), created
# with --system-site-packages so it inherits the container's torch instead
# of pip re-downloading a possibly-mismatched build. Nothing is installed
# with `pip install --user` (that would land in $HOME and leak into every
# other apptainer run you do) or into the container's own site-packages
# (the container is read-only anyway).

CONTAINER=/projects/academic/alipour/payamabd/pytorch_ngc_25.02.sif
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where the isolated venv lives. Override by exporting MONET_VENV_DIR before
# sourcing this file if you'd rather keep it somewhere other than alongside
# the repo (e.g. if $REPO_DIR is on a filesystem compute nodes can't write
# to, or you want it under /projects/... instead).
VENV_DIR="${MONET_VENV_DIR:-${REPO_DIR}/.venv-apptainer}"

# Extend/replace this if your dataset or model live outside $REPO_DIR's
# parents (the pytorch_ngc container binds $HOME automatically; add more
# --bind flags here if e.g. Monet-SFT-125K and the Qwen3-VL-2B-Instruct
# checkpoint live under a different /projects/... path than the .sif itself).
APPTAINER_BINDS="--bind /projects/academic/alipour:/projects/academic/alipour"

VENV_PY="${VENV_DIR}/bin/python"

# Runs `python -m torch.distributed.run` (what the `torchrun` console script
# itself wraps) with the venv's own interpreter -- calling the container's
# global `torchrun` directly would use the container's system python via its
# shebang, which can't see the packages installed into $VENV_DIR.
run_group () {
  # $1 = het-group index, $2 = GPUs on that node, $3 = rdzv id (must match
  # across both groups of the same phase, and differ between phases so their
  # rendezvous rounds can't collide), $4 = MASTER_ADDR:MASTER_PORT, remaining
  # args = the python -m module + its arguments.
  local het_group=$1 nproc=$2 rdzv_id=$3 rdzv_endpoint=$4
  shift 4
  if [ ! -x "${VENV_PY}" ]; then
    echo "No venv at ${VENV_DIR} -- run script_examples/setup_env.sh first (on a node with internet access)." >&2
    exit 1
  fi
  srun --het-group="${het_group}" apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" \
    "${VENV_PY}" -m torch.distributed.run \
      --nnodes=2 --nproc-per-node="${nproc}" \
      --rdzv-id="${rdzv_id}" --rdzv-backend=c10d --rdzv-endpoint="${rdzv_endpoint}" \
      "$@"
}
