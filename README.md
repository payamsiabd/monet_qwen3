# Monet for Qwen3-VL-2B-Instruct

This repository reproduces **SFT Stage 1, 2, and 3** of
[Monet: Reasoning in Latent Visual Space Beyond Images and Language](http://arxiv.org/abs/2511.21395)
(CVPR 2026) for **Qwen3-VL-2B-Instruct**, ported from the
[official Monet repo](https://github.com/NOVAglow646/Monet) (which targets Qwen2.5-VL-7B-Instruct).

## Scope

Only the three-stage SFT pipeline (`Monet-SFT`) is ported here — the part of the paper that
produces `M_SFT`. The RL stage (VLPO) and the vLLM-based inference/serving code are **not**
included; see [Not ported](#not-ported) below.

## What changed vs. the original repo

The goal was to change as little as possible. Concretely:

- **`monet_qwen_model/modeling_qwen3_vl_monet.py`** (new) replaces
  `monet_qwen_model/modeling_qwen2_5_vl_monet.py`. It is the same kind of patch — a
  drop-in replacement for `transformers.models.qwen3_vl.modeling_qwen3_vl` — carrying
  the same latent-reasoning forward pass (`latent_mode` segment-by-segment decoding,
  `ce_patch_pos`/`ce_patch_vec` scatter-back, the weighted CE loss, the
  all-layers cosine-similarity alignment loss, and the controlled 4D attention flow).
  It was derived by diffing the reference file against vanilla
  `transformers==4.54.0`'s `modeling_qwen2_5_vl.py` to isolate Monet's actual patch,
  then re-applying the equivalent changes to vanilla `transformers==4.57.1`'s
  `modeling_qwen3_vl.py`. Two points needed real adaptation, not just renaming:
  - **DeepStack.** Qwen3-VL injects vision features into the first few decoder layers
    (`Qwen3VLTextModel._deepstack_process`), not only into the input embeddings like
    Qwen2.5-VL. Monet's segment-by-segment latent forward calls the language model many
    times over slices of the sequence, so each slice now carries its own slice of the
    DeepStack features (see `_slice_deepstack` in `Qwen3VLModel.forward`) — otherwise
    auxiliary/question images would silently lose part of their visual grounding.
  - **`@check_model_inputs`.** Recent `transformers` records `hidden_states`/`attentions`
    by monkey-patching submodules whose *class identity* matches
    `Qwen3VLPreTrainedModel._can_record_outputs`. Because of this, the vision tower and
    text decoder stack (`Qwen3VLVisionModel`, `Qwen3VLTextModel`, and everything they're
    built from) are **imported unchanged** from the original module rather than
    redefined — redefining them under the same names would silently break
    `output_hidden_states=True`, which the alignment losses in Stage 2/3 depend on. The
    two outer forwards Monet actually changes (`Qwen3VLModel`,
    `Qwen3VLForConditionalGeneration`) drop `@check_model_inputs` and manage
    `hidden_states`/`ce_patch_*`/`alignment_loss` manually instead, the same way the
    reference Qwen2.5-VL patch already did (that file predates the decorator).
  - The attention-internals dead code from the reference patch (`attn_loss`,
    `collect_emphasize_attn`, the affine-subspace-alignment exploration in Sec. 4.3 of
    the paper) is **not** ported, for the same reason — it lives inside
    `Qwen3VLTextAttention`, which is now imported rather than redefined. It was never
    wired into `src/trainer.py`'s loss in the original repo either, so this drops no
    functionality; `alignment_loss` (the one such helper that *is* used) is kept.
- **`monet_qwen_model/apply_qwen3_vl_monet.py`** (new) mirrors
  `apply_qwen2_5_monet.py`'s `sys.modules` swap, targeting
  `transformers.models.qwen3_vl.modeling_qwen3_vl` instead.
- **`src/main.py`, `src/precompute_teacher_reps.py`, `src/precompute_teacher_latents.py`**:
  only the model imports changed (`Qwen2_5_VLForConditionalGeneration`/`Qwen2_5_VLConfig`
  → `Qwen3VLForConditionalGeneration`/`Qwen3VLConfig`, `apply_qwen2_5_monet` →
  `apply_qwen3_vl_monet`). `main.py` additionally resizes the token embedding table if
  the 5 tokens Monet adds (`<abs_vis_token>`, `<abs_vis_token_pad>`, `</abs_vis_token>`,
  `<observation>`, `</observation>`) don't fit in Qwen3-VL's tokenizer — the original
  repo didn't need this because Qwen2.5-VL-7B-Instruct's tokenizer happens to have
  enough reserved slots; the resize is a no-op if Qwen3-VL's does too.
- **`src/task.py`, `src/trainer.py`, `src/utils.py`**: byte-for-byte unchanged. All the
  data preprocessing, the 4D controlled-attention mask construction, and the three
  `CustomTrainerSFT_STAGE{1,2,3}` loss computations are architecture-agnostic (they only
  touch `input_ids`/tensors and the model's `forward(...)` kwargs contract), so nothing
  here needed to change.
- **`script_examples/sft_stage{1,2,3}.sh`**: `--load_model_path` now points at
  `Qwen3-VL-2B-Instruct`; two bugs in the original `sft_stage3.sh` are fixed
  (`--stage "avt_v5_stage2"` → `--stage "sft_stage3"`, and a missing `/` in
  `sft_stage1${STAGE1_MODEL}` → `sft_stage1/${STAGE1_MODEL}`) since otherwise the
  script cannot run at all. All `-m src.main`/`-m src.precompute_*` hyperparameters
  (epochs, batch size, learning rate, latent size, alignment weight, ...) are left
  exactly as in the original scripts. The launch mechanics around that command *are*
  changed: the original single-node `conda activate` + `torchrun --nproc-per-node=8`
  is replaced with SLURM `sbatch` + `apptainer exec --nv` + a heterogeneous multi-node
  `torchrun` launch matched to this fork's actual cluster (5 L40S GPUs split 2+3 across
  two asymmetric nodes) — see [SLURM + apptainer](#slurm--apptainer-multi-node-setup)
  below for what that involves and why.
- **`requirements.txt`**: `transformers==4.54.0` → `4.57.1` (Qwen3-VL support was only
  added in `4.57.0`, which PyPI yanked; `4.57.1` is the first usable release) and
  `trl==0.15.2` → `0.24.0` (the closest release declaring `transformers>=4.56.1`).
  `vllm` is dropped since it's only needed for the RL/inference stages this fork
  doesn't cover. TRL's `SFTTrainer` internals move fast; if a newer/older `trl` breaks
  against the `compute_loss` override in `src/trainer.py`, that's the first thing to
  pin differently.
- **`deepspeed/ds_zero2_gpu.json`**: unchanged.

## Not ported

- **RL / VLPO** (`RL/`, `script_examples`'s RL script) and the **vLLM inference runner**
  (`inference/`) from the original repo — the paper's Stage 4 and evaluation setup,
  respectively. Out of scope: the user request was specifically stages 1–3.
- The dead attention-analysis code discussed above.

## Verifying the port

This environment has no GPU and no model/data downloaded, so the port could not be
trained end-to-end here. What *was* verified on CPU, with a tiny randomly-initialized
Qwen3-VL config (small vision tower with 2 blocks/1 DeepStack layer, small text decoder
with GQA), is that the full latent-reasoning control flow runs and produces finite
losses and gradients:

- A plain CE forward with a question image (Stage-1 shape).
- A `latent_mode=True` forward (question image + auxiliary image + latent placeholders,
  4D controlled attention) followed by a `latent_mode=False` CE+alignment forward using
  the collected `ce_patch_pos`/`ce_patch_vec`, matching `CustomTrainerSFT_STAGE2`
  (Stage-2 shape).
- The same two-forward pattern with per-latent-step alignment against precomputed
  target latents, matching `CustomTrainerSFT_STAGE3` (Stage-3 shape).
- A full backward pass through all of the above, including the latent-only
  backpropagation trick (`compute_latents_only_loss` in `src/trainer.py`) — confirmed
  gradients reach every trainable (non-vision) parameter.

This exercises the DeepStack-aware segment loop, the controlled attention mask, the
weighted CE loss, and the alignment loss end to end, but it is **not** a substitute for
an actual training run with real data and a real checkpoint — please sanity-check loss
curves and a few generations after Stage 1 before investing in Stages 2 and 3.

## Usage

Same as the [original repo](https://github.com/NOVAglow646/Monet#-sft-training):

```bash
pip install -r requirements.txt
```

1. Download [Monet-SFT-125K](https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K)
   and a local copy of `Qwen/Qwen3-VL-2B-Instruct`.
2. Fill in the `path_to_your_dataset` / `path_to_your_model` placeholders in
   `script_examples/sft_stage1.sh`, `sft_stage2.sh`, `sft_stage3.sh` and run them in
   order (Stage 2 and Stage 3 each first precompute teacher representations/latents,
   then train).

## SLURM + apptainer (multi-node) setup

`script_examples/sft_stage{1,2,3}.sh` are `sbatch`-submittable directly
(`sbatch script_examples/sft_stage1.sh`, etc.) and are set up for a specific
asymmetric two-node cluster: 5 total L40S GPUs split 2 + 3 across two nodes,
run inside an NGC PyTorch apptainer image
(`/projects/academic/alipour/payamabd/pytorch_ngc_25.02.sif`).

- A plain `sbatch --nodes=2 --gres=gpu:...:N` job can only request the *same*
  N GPUs on every node, so it tops out at 2+2=4 GPUs here. Getting the full
  2+3=5 needs a **SLURM heterogeneous job** — each script has two
  `#SBATCH`/`#SBATCH hetjob`-separated resource blocks (one per node's GPU
  count), and launches torch's distributed runner on each side via
  `srun --het-group=<i>`, both pointing at the same c10d rendezvous endpoint
  so they join one process group.
- **Environment**: one isolated virtualenv, built *inside* the container so
  it inherits the container's own CUDA-matched `torch`/`torchvision` (via
  `python -m venv --system-site-packages`) without touching the container
  itself (read-only anyway) or `$HOME` (no `pip install --user`). Build it
  once — this needs PyPI access, which most SLURM clusters only give
  login/data-transfer nodes, not compute nodes, so run it directly rather
  than through `sbatch`:
  ```bash
  bash script_examples/setup_env.sh
  ```
  This installs the single `requirements.txt` (unchanged, still includes
  bare `torch`/`torchvision`) into the venv; since those two are unpinned
  and already importable via the container's system site-packages, pip
  leaves them alone and only installs what's actually missing
  (`transformers`, `trl`, `deepspeed`, ...) — see `script_examples/
  cluster_env.sh` for the shared config (container path, venv location,
  bind mounts) both `setup_env.sh` and the three stage scripts source.
- Each stage script's header comment explains the resource requests, the
  `--nodelist` override if you want to pin the exact nodes from `scontrol
  show node` rather than let SLURM pick any node with matching GRES in the
  `alipour` partition, and the `NCCL_SOCKET_IFNAME`/`NCCL_DEBUG` knobs to set
  if the job hangs at rendezvous (by far the most common first failure on a
  new multi-node setup — it's almost always NCCL picking the wrong network
  interface between nodes).
- **These sbatch scripts were written and reviewed but not run** — this
  environment has no SLURM/GPU/apptainer access. Treat the resource requests
  (CPUs/memory per node) as reasonable starting points based on the
  `scontrol show node` snapshot at the time, not tuned numbers; adjust with
  `squeue`/`sinfo` for current load, and expect to iterate on the NCCL/
  rendezvous settings once you have real multi-node logs to look at.

See `monet_qwen_model/modeling_qwen3_vl_monet.py`'s module docstring-style comments and
`Qwen3VLModel.forward`/`Qwen3VLForConditionalGeneration.forward` for the implementation
of the forward process with latent embeddings (mirroring the original repo's README
pointer to `Qwen2_5_VLModel:forward` / `Qwen2_5_VLForConditionalGeneration:forward`).
