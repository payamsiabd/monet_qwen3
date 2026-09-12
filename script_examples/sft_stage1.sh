#!/bin/bash
# ============================================================================
# SLURM heterogeneous multi-node launcher for SFT Stage 1, submit with:
#   sbatch script_examples/sft_stage1.sh
#
# Targets 5 L40S GPUs split across two asymmetric nodes (2 + 3 GPUs), e.g.
#   cpn-f07-17: 2x L40S     cpn-d04-11: 3x L40S
# A plain (non-heterogeneous) `sbatch --nodes=2 --gres=gpu:...:N` job can only
# request the SAME N on every node, so it tops out at 2+2=4 GPUs here. Getting
# the full 2+3=5 requires a SLURM heterogeneous job (the two "#SBATCH hetjob"-
# separated blocks below), with one `srun --het-group=<i>` launching torchrun
# on each side, both pointing at the same c10d rendezvous endpoint.
#
# This was written and reviewed, but NOT run on your cluster (no SLURM/GPU/
# apptainer access from the environment that produced it). The most likely
# failure points if the job hangs or errors are noted inline below.
# ============================================================================
#SBATCH --job-name=monet_sft_stage1
#SBATCH --partition=alipour
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH hetjob
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:nvidia_L40S:2 --cpus-per-task=8 --mem=64G
#SBATCH hetjob
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:nvidia_L40S:3 --cpus-per-task=12 --mem=96G
# To pin to the exact nodes from `scontrol show node` instead of letting SLURM
# pick any node with matching GRES in the partition, uncomment (one per het
# job component, i.e. right after each of the two blocks above):
#   #SBATCH --nodelist=cpn-f07-17     (for the 2-GPU component)
#   #SBATCH --nodelist=cpn-d04-11     (for the 3-GPU component)

set -euo pipefail
mkdir -p logs

CONTAINER=/projects/academic/alipour/payamabd/pytorch_ngc_25.02.sif
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Extend/replace this if your dataset or model live outside $REPO_DIR's parents
# (the pytorch_ngc container binds $HOME automatically; add more --bind flags
# below if e.g. Monet-SFT-125K and the Qwen3-VL-2B-Instruct checkpoint live
# under a different /projects/... path than the .sif itself).
APPTAINER_BINDS="--bind /projects/academic/alipour:/projects/academic/alipour"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
MASTER_PORT=29501
RDZV_ID="${SLURM_JOB_ID}_stage1"

# If NCCL hangs at rendezvous/init with two nodes (the most common first
# failure on a new cluster), it's almost always the network interface NCCL
# picks. Uncomment and set to whatever `ip addr` shows as the interface that
# can reach both nodes (often ib0 for InfiniBand, or the ethernet name on
# pure-Ethernet clusters); NCCL_DEBUG=INFO on both ranks will show which
# interface it tried.
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_DEBUG=INFO

CE_EMPHASIZE_FACTOR=2.0
SAVE_CKPT=sft_stage1_ce${CE_EMPHASIZE_FACTOR}
TRAIN_ARGS=(
  --epochs 4
  --bsz 1
  --grad_accum_steps 16
  --stage "sft_stage1"
  --data_path
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json"
    "path_to_your_dataset/Monet-SFT-125K/CogCoM/train.json"
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
  --load_model_path path_to_your_model/Qwen3-VL-2B-Instruct
  --save_model_path path_to_your_model/Monet_checkpoints/sft_stage1/${SAVE_CKPT}
  --dataset_root path_to_your_dataset/Monet-SFT-125K
  --deepspeed ./deepspeed/ds_zero2_gpu.json
  --wandb_name ${SAVE_CKPT}
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}
)

run_group () {
  local het_group=$1 nproc=$2
  srun --het-group="${het_group}" apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" \
    torchrun \
      --nnodes=2 --nproc-per-node="${nproc}" \
      --rdzv-id="${RDZV_ID}" --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      -m src.main "${TRAIN_ARGS[@]}"
}

cd "$REPO_DIR"
run_group 0 2 &
run_group 1 3 &
wait
