#!/bin/bash
# ============================================================================
# SLURM heterogeneous multi-node launcher for SFT Stage 3, submit with:
#   sbatch script_examples/sft_stage3.sh
#
# Runs two sequential phases inside one job allocation: (1) precompute the
# target latent embeddings with the Stage-2 checkpoint, then (2) SFT Stage 3
# training against them. Both phases use the same 2+3=5 GPU heterogeneous
# split across two nodes as sft_stage1.sh -- see that file's header comment
# for why a plain `sbatch --nodes=2 --gres=...` can't express an asymmetric
# 2+3 split, and for the NCCL troubleshooting note if the job hangs at
# rendezvous.
#
# This was written and reviewed, but NOT run on your cluster (no SLURM/GPU/
# apptainer access from the environment that produced it).
# ============================================================================
#SBATCH --job-name=monet_sft_stage3
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
# job component, right after each of the two blocks above):
#   #SBATCH --nodelist=cpn-f07-17     (for the 2-GPU component)
#   #SBATCH --nodelist=cpn-d04-11     (for the 3-GPU component)

set -euo pipefail
mkdir -p logs

CONTAINER=/projects/academic/alipour/payamabd/pytorch_ngc_25.02.sif
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPTAINER_BINDS="--bind /projects/academic/alipour:/projects/academic/alipour"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
MASTER_PORT=29503

# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_DEBUG=INFO

run_group () {
  # $1 = het-group index, $2 = GPUs on that node, $3 = rdzv id (must match
  # across both groups of the same phase, and differ between the two phases
  # below so their rendezvous rounds can't collide), remaining args = the
  # python -m module + its arguments.
  local het_group=$1 nproc=$2 rdzv_id=$3
  shift 3
  srun --het-group="${het_group}" apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" \
    torchrun \
      --nnodes=2 --nproc-per-node="${nproc}" \
      --rdzv-id="${rdzv_id}" --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      "$@"
}

cd "$REPO_DIR"

# ----------------------------------------------------------------------------
# STEP 1: precompute target latent embeddings using the model trained in SFT
# Stage 2
# ----------------------------------------------------------------------------
STAGE1_MODEL=sft_stage1_ce2.0
TEACHER_LATENT_SIZE=8
TEACHER_CE_EMPHASIZE_FACTOR=4.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER=sft_stage2_latent${TEACHER_LATENT_SIZE}_ce${TEACHER_CE_EMPHASIZE_FACTOR}_al${TEACHER_ALIGN_WEIGHT}_emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}
PRECOMPUTE_ARGS=(
  -m src.precompute_teacher_latents
  --bsz 1
  --data_path
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json"
    "path_to_your_dataset/Monet-SFT-125K/CogCoM/train.json"
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
  --load_model_path path_to_your_model/Monet_checkpoints/sft_stage2/${TEACHER}
  --save_model_path path_to_your_model/Monet_checkpoints/monet_precomputed_target_latent/${TEACHER}
  --dataset_root path_to_your_dataset
  --deepspeed ./deepspeed/ds_zero2_gpu.json
  --latent_size ${TEACHER_LATENT_SIZE}
  --output_hidden_states
  --resume
)
run_group 0 2 "${SLURM_JOB_ID}_stage3_precompute" "${PRECOMPUTE_ARGS[@]}" &
run_group 1 3 "${SLURM_JOB_ID}_stage3_precompute" "${PRECOMPUTE_ARGS[@]}" &
wait

# ----------------------------------------------------------------------------
# STEP 2: SFT stage3 training
# ----------------------------------------------------------------------------
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=sft_stage3_target-latent${TEACHER_LATENT_SIZE}-al${TEACHER_ALIGN_WEIGHT}-emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}_student-latent${LATENT_SIZE}-ce${CE_EMPHASIZE_FACTOR}-al${ALIGNMENT_WEIGHT}-emph${EMPHASIZE_LATENT_WEIGHT}
TRAIN_ARGS=(
  -m src.main
  --epochs 2
  --bsz 1
  --grad_accum_steps 16
  --stage "sft_stage3"
  --data_path
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json"
    "path_to_your_dataset/Monet-SFT-125K/CogCoM/train.json"
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json"
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json"
  --log_file "./log.txt"
  --load_model_path path_to_your_model/Monet_checkpoints/sft_stage1/${STAGE1_MODEL}
  --save_model_path path_to_your_model/Monet_checkpoints/sft_stage3/${SAVE_CKPT}
  --dataset_root path_to_your_dataset/Monet-SFT-125K
  --deepspeed ./deepspeed/ds_zero2_gpu.json
  --wandb_name ${SAVE_CKPT}
  --latent_size ${LATENT_SIZE}
  --alignment_weight ${ALIGNMENT_WEIGHT}
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}
  --teacher_latent_dir path_to_your_model/Monet_checkpoints/monet_precomputed_target_latent/${TEACHER}
  --alignment_layer all_layers
)
run_group 0 2 "${SLURM_JOB_ID}_stage3_train" "${TRAIN_ARGS[@]}" &
run_group 1 3 "${SLURM_JOB_ID}_stage3_train" "${TRAIN_ARGS[@]}" &
wait
