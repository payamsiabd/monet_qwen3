from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel


# ============================================================================
# Monet: dead-code exploration helpers ported verbatim from the reference
# Qwen2.5-VL patch (monet_qwen_model/modeling_qwen2_5_vl_monet.py in the
# original Monet repo). Not wired into src/trainer.py's loss computation
# (the paper reports these as unsuccessful early attempts, see Sec 4.3);
# kept here only for fidelity to the reference implementation.
# ============================================================================


def compute_affine_projectors(
    inputs_embeds: torch.Tensor,
    seg: List[List[Tuple]],
    *,
    stop_grad_img_subspace: bool = True,
    solver: str = "svd",            # "svd" | "qr"
    rank_eps: float = 50,
) -> List[List[Dict[str, Any]]]:
    """
    Compute per-image affine subspace projectors in an implicit form (mu, U) so that
    the projector P can be applied as P(x - mu) = U @ (U^T @ (x - mu)).
    We return a nested list aligned as projectors[b][t_non_question] for each batch b.
    """
    assert inputs_embeds.dim() == 3, "inputs_embeds must be [B, S, d]"
    B, S, d = inputs_embeds.shape
    device, dtype = inputs_embeds.device, inputs_embeds.dtype

    def _to_index_tensor(pos) -> torch.LongTensor:
        if torch.is_tensor(pos):
            return pos.to(device=device, dtype=torch.long)
        elif isinstance(pos, (list, tuple)):
            return torch.tensor(list(pos), device=device, dtype=torch.long)
        else:
            raise TypeError("Invalid seg position container; expect tensor/list/tuple of ints.")

    projectors: List[List[Dict[str, Any]]] = []
    for b in range(B):
        seg_b = seg[b] if b < len(seg) else []
        if len(seg_b) == 0:
            projectors.append([])
            continue
        proj_b: List[Dict[str, Any]] = []

        for s_idx in range(len(seg_b)):
            pos_container = seg_b[s_idx][0]  # positions for this image-step
            if pos_container is None:
                proj_b.append({"mu": None, "U": None, "idx": None, "valid": False})
                continue
            idx = _to_index_tensor(pos_container)
            if idx.numel() == 0:
                proj_b.append({"mu": None, "U": None, "idx": idx, "valid": False})
                continue

            # Image token embeddings [k, d]
            Z_img = inputs_embeds[b, idx, :]  # [k, d]

            # Affine centering: mu and centered matrix X = Z_img - mu
            if stop_grad_img_subspace:
                mu = Z_img.detach().mean(dim=0)         # [d]
                X = (Z_img.detach() - mu)               # [k, d]
            else:
                mu = Z_img.mean(dim=0)
                X = Z_img - mu

            # Handle degenerate case quickly
            if X.numel() == 0 or torch.allclose(X, torch.zeros_like(X), atol=1e-12, rtol=0.0):
                U = torch.empty(d, 0, device=device, dtype=dtype)
                proj_b.append({"mu": mu, "U": U, "idx": idx, "valid": True})
                continue

            # Compute an orthonormal basis U for span(X) in R^d
            if solver.lower() == "qr":
                Qt, Rt = torch.linalg.qr(X.transpose(0, 1).to(torch.float32), mode='reduced')  # Qt:[d, r], Rt:[r, k]
                diag = torch.abs(torch.diagonal(Rt))
                if diag.numel() > 0:
                    thr = rank_eps * float(diag.max().item())
                    r_eff = int((diag > thr).sum().item())
                else:
                    r_eff = 0
                U = (Qt[:, :r_eff] if r_eff > 0 else Qt[:, :0]).to(dtype)  # [d, r_eff]
            else:
                Ux, Sx, Vh = torch.linalg.svd(X.to(torch.float32), full_matrices=False)  # Ux:[k,r], Sx:[r], Vh:[r,d]
                if Sx.numel() > 0:
                    thr = rank_eps * float(Sx.max().item())
                    keep = (Sx > thr)
                    if keep.any():
                        V = Vh[keep, :].transpose(0, 1)  # [d, r_eff]
                    else:
                        V = Vh[:0, :].transpose(0, 1)    # [d, 0]
                else:
                    V = Vh.transpose(0, 1)               # [d, 0]
                U = V.to(dtype)                           # [d, r_eff]

            proj_b.append({"mu": mu, "U": U, "idx": idx, "valid": True})

        projectors.append(proj_b)

    return projectors


def affine_subspace_alignment_loss(
    projectors: List[List[Dict[str, Any]]],
    ce_patch_vec: List[torch.Tensor],
    *,
    latent_size: int,
    reduction: str = "mean",   # "mean" | "sum"
) -> torch.Tensor:
    """
    Affine-subspace alignment loss for flattened latent tensors.
    """
    assert isinstance(projectors, list) and isinstance(ce_patch_vec, (list, tuple)), \
        "projectors and ce_patch_vec must be lists over batch."
    assert isinstance(latent_size, int) and latent_size > 0, "latent_size must be a positive int."

    loss_terms: List[torch.Tensor] = []
    device, dtype = None, None

    for b, projs_b in enumerate(projectors):
        if len(projs_b) == 0:
            continue

        Zflat = ce_patch_vec[b]
        assert torch.is_tensor(Zflat) and Zflat.dim() == 2, \
            f"ce_patch_vec[{b}] must be 2D [latent_size*t, d]; got shape {tuple(Zflat.shape)}"

        if device is None:
            device, dtype = Zflat.device, Zflat.dtype

        N, d = Zflat.shape
        assert N % latent_size == 0, \
            f"First dim {N} is not divisible by latent_size {latent_size}."
        t = N // latent_size
        assert t == len(projs_b), \
            f"Time steps mismatch for batch {b}: latents imply t={t}, but projectors has {len(projs_b)}."

        Zsteps = Zflat.reshape(t, latent_size, d)

        for s in range(t):
            proj = projs_b[s]
            mu = proj.get("mu", None)
            U = proj.get("U", None)
            valid = bool(proj.get("valid", False))

            Z_lat = Zsteps[s]  # [latent_size, d]

            if (not valid) or (mu is None) or (U is None):
                res2 = (Z_lat * Z_lat).sum(dim=-1)  # [latent_size]
                loss_terms.append(res2.mean())
                continue

            Zc = (Z_lat - mu)  # [latent_size, d]

            if U.numel() == 0 or U.shape[1] == 0:
                res2 = (Zc * Zc).sum(dim=-1)  # [latent_size]
                loss_terms.append(res2.mean())
                continue

            U32 = U.to(torch.float32)          # [d, r]
            Zc32 = Zc.to(torch.float32)         # [latent_size, d]
            coeff = Zc32 @ U32                  # [latent_size, r]
            proj_ = coeff @ U32.transpose(0, 1)  # [latent_size, d]
            resid = Zc32 - proj_                 # [latent_size, d]
            res2 = (resid * resid).sum(dim=-1)  # [latent_size]
            loss_terms.append(res2.mean().to(Zc.dtype))

    if len(loss_terms) == 0:
        if device is None:
            device, dtype = torch.device("cpu"), torch.float32
        return torch.zeros((), device=device, dtype=dtype)

    loss_stack = torch.stack(loss_terms)
    if reduction == "sum":
        return loss_stack.sum()
    else:
        return loss_stack.mean()


def alignment_loss(teacher_hidden_states: torch.Tensor, student_hidden_states: Union[List[torch.Tensor], torch.Tensor]):
    total_loss = 0
    if teacher_hidden_states.dim() == 3:  # [num_layer, num_align_in_a_seg, dim], align all layers
        total_loss = (1 - torch.nn.functional.cosine_similarity(teacher_hidden_states.to(student_hidden_states.device), student_hidden_states)).mean()
    elif teacher_hidden_states.dim() == 1:  # align last layer
        total_loss = 1 - torch.nn.functional.cosine_similarity(student_hidden_states, teacher_hidden_states, 0)
    return total_loss


@torch.no_grad()
def _svd_select_U(
    X: torch.Tensor,
    r_max: int,
    rank_eps: float,
    energy_keep: Optional[float] = None,
) -> torch.Tensor:
    """
    Select an orthonormal basis U (d x r) for span(X) using truncated SVD with rank/energy constraints.
    """
    Ux, S, Vh = torch.linalg.svd(X, full_matrices=False)  # Ux: [k, r], S: [r], Vh: [r, d]
    if S.numel() == 0:
        return Vh.transpose(0, 1)[:, :0]  # [d, 0]

    thr = float(S.max().item()) * rank_eps
    keep_mag = (S > thr)

    if energy_keep is not None and energy_keep > 0.0 and energy_keep <= 1.0:
        s2 = S * S
        cum = torch.cumsum(s2, dim=0)
        total = s2.sum()
        r_energy = int((cum / (total + 1e-12) >= energy_keep).nonzero(as_tuple=False)[0].item()) + 1
    else:
        r_energy = S.numel()

    r_from_eps = int(keep_mag.sum().item())
    r = min(r_from_eps, r_energy, int(r_max))
    if r <= 0:
        return Vh.transpose(0, 1)[:, :0]  # [d, 0]
    return Vh[:r, :].transpose(0, 1)  # [d, r]


# Reuse the vanilla Qwen3-VL vision tower and text decoder stack unmodified:
# Monet only changes the multimodal fusion / latent-reasoning forward
# (Qwen3VLModel, Qwen3VLForConditionalGeneration below) and the two output
# dataclasses. Keeping these as the *same* class objects as the original
# module (rather than redefining them here) matters: transformers'
# @check_model_inputs decorator records hidden_states/attentions by
# monkey-patching submodules whose *class identity* matches
# Qwen3VLPreTrainedModel._can_record_outputs (Qwen3VLTextDecoderLayer /
# Qwen3VLTextAttention) - a locally-redefined class of the same name would
# not match, silently breaking hidden_states capture.
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLTextModel


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen3VLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    ce_patch_pos: Optional[List[List[int]]] = None
    ce_patch_vec: Optional[List[torch.Tensor]] = None
    alignment_loss: Optional[torch.FloatTensor] = None


@auto_docstring
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        latent_mode: Optional[bool] = False,
        alignment_poss: Optional[List[List]] = None,
        teacher_hidden_states_for_alignment: Optional[Union[List[List[torch.Tensor]], torch.Tensor]] = None,  # for the latent forward
        ce_patch_pos: Optional[List[List[int]]] = None,
        ce_patch_vec: Optional[List[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        use_cache = use_cache if use_cache is not None else self.config.get_text_config().use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ------------------------------------------------------------------
        # 0. Basic shapes & containers
        # ------------------------------------------------------------------
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Qwen3-VL injects encoder features into the first few decoder layers
        # (DeepStack, see Qwen3VLTextModel._deepstack_process) in addition to the
        # input-embedding scatter above. Monet's segment-by-segment latent forward
        # below calls `self.language_model` many times over slices of the sequence,
        # so we precompute a full-length (per-position) DeepStack tensor here and
        # re-slice+re-compact a (visual_pos_masks, deepstack_visual_embeds) pair for
        # any sub-segment, to keep this behavior faithful to vanilla Qwen3-VL.
        has_deepstack = deepstack_visual_embeds is not None and len(deepstack_visual_embeds) > 0
        full_deepstack_embeds = None
        if has_deepstack:
            full_deepstack_embeds = []
            for ds_embed in deepstack_visual_embeds:
                full_ds = inputs_embeds.new_zeros(batch_size, seq_len, ds_embed.shape[-1])
                full_ds[visual_pos_masks] = ds_embed.to(dtype)
                full_deepstack_embeds.append(full_ds)

        def _slice_deepstack(b: int, s: int, e: int):
            if not has_deepstack:
                return None, None
            vp = visual_pos_masks[b : b + 1, s:e]
            if not bool(vp.any()):
                return None, None
            ds = [full_deepstack_embeds[l][b : b + 1, s:e][vp] for l in range(len(full_deepstack_embeds))]
            return vp, ds

        attn_mask_4d_full = attention_mask_4d['full_attention'] if attention_mask_4d is not None else None

        if latent_mode:  # latent forward, SFT Stage 2 student CoT or SFT Stage 3
            ce_patch_pos = [[] for _ in range(batch_size)]   # List[List[int]]
            ce_patch_vec = [[] for _ in range(batch_size)]   # List[List[Tensor(H,)]]
            hidden_states_to_return = [[] for _ in range(batch_size)]
            total_align_loss = None

            def run_segment_forward(language_model,
                        seg_embeds, seg_pos_ids, seg_att_m,
                        past_kv,
                        need_hidden: bool,
                        no_grad_mode: bool,
                        seg_visual_pos_masks=None,
                        seg_deepstack_visual_embeds=None,
                        **kwargs):
                '''
                Run a forward of a segment through the LLM.
                '''
                ctx = torch.inference_mode() if no_grad_mode else nullcontext()

                with ctx:
                    out = language_model(
                        input_ids=None,
                        inputs_embeds=seg_embeds,
                        position_ids=seg_pos_ids,
                        attention_mask=seg_att_m,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=need_hidden and (not no_grad_mode),
                        visual_pos_masks=seg_visual_pos_masks,
                        deepstack_visual_embeds=seg_deepstack_visual_embeds,
                        return_dict=True,
                        **kwargs,
                    )
                return out

            def _first_pattern_end(row_ids: torch.LongTensor, pattern_ids: List[int]) -> int:
                pattern_ids = torch.tensor(pattern_ids, dtype=row_ids.dtype, device=row_ids.device)
                wins = row_ids.unfold(0, pattern_ids.numel(), 1)  # [L-M+1, M]
                eq = (wins == pattern_ids).all(dim=-1)
                idx = torch.nonzero(eq, as_tuple=False)
                if idx.numel() == 0:
                    return 0
                return int(idx[0].item() + pattern_ids.numel())

            batch_last_hidden_state = inputs_embeds.new_zeros(batch_size, seq_len, hidden_dim)
            past_key_values_batch = [None] * batch_size

            # ------------------------------------------------------------------
            # 1.  Build latent-token position lists
            # ------------------------------------------------------------------
            latent_lists = [
                (input_ids[b] == self.config.latent_token_id).nonzero(as_tuple=False).flatten().tolist()
                for b in range(batch_size)
            ]

            if alignment_poss is None:
                alignment_poss = [[] for _ in range(batch_size)]

            # ------------------------------------------------------------------
            # 2.  Iterate samples.
            # WARNING: we only use batch_size=1 in our training. Larger batch_size has not been tested.
            # ------------------------------------------------------------------
            for b in range(batch_size):
                latent_pos = latent_lists[b]  # List[int], positions of latent tokens in this sample
                align_pos = sorted(alignment_poss[b])
                align_ptr = 0  # pointer into align_pos list

                # find the start of responses
                ans_start = _first_pattern_end(input_ids[b], self.config.answer_start_pattern)

                # `teacher_hidden_states_for_alignment` corresponds to the teacher hidden states of the observation tokens in SFT Stage 2, or the target hidden states of the latent tokens in SFT Stage 3
                if teacher_hidden_states_for_alignment is not None:
                    assert teacher_hidden_states_for_alignment[b].shape[1] == len(align_pos), f"teacher_hidden_states_for_alignment.shape[1] ({teacher_hidden_states_for_alignment[b].shape[1]}) != len(align_pos) ({len(align_pos)})"

                seq_embeds = inputs_embeds[b : b + 1]           # (1,L,H)
                pos_ids_s = position_ids[:, b : b + 1]           # (3,1,L)
                if attn_mask_4d_full is not None:
                    attn_mask_s = attn_mask_4d_full[b : b + 1]   # (1,1,L,L)
                else:
                    attn_mask_s = attention_mask[b : b + 1]      # (1,L)

                # ------------------------------------------------------------------
                # 2.1  Forward the pre-answer segment
                # ------------------------------------------------------------------
                past_kv = None
                if ans_start > 0:  # Forward the pre-answer segment
                    pre_embeds = seq_embeds[:, :ans_start, :]
                    pre_pos_ids = pos_ids_s[:, :, :ans_start]
                    pre_att_m = (
                        attn_mask_s[:, :, :ans_start, :ans_start] if attn_mask_4d_full is not None
                        else attn_mask_s[:, :ans_start]
                    )
                    pre_vp, pre_ds = _slice_deepstack(b, 0, ans_start)
                    pre_out = run_segment_forward(
                        language_model=self.language_model,
                        seg_embeds=pre_embeds,
                        seg_pos_ids=pre_pos_ids,
                        seg_att_m=pre_att_m,
                        past_kv=None,
                        need_hidden=False,
                        no_grad_mode=False,
                        seg_visual_pos_masks=pre_vp,
                        seg_deepstack_visual_embeds=pre_ds,
                        **kwargs,
                    )
                    batch_last_hidden_state[b, :ans_start, :] = pre_out.last_hidden_state[0]
                    past_kv = pre_out.past_key_values

                # ------------------------------------------------------------------
                # 2.2 Forward the non-latent segments (text and image tokens) & latent tokens
                # ------------------------------------------------------------------
                align_ptr = 0
                align_losses_this_b = 0.
                prev_idx = ans_start
                image_mask = input_ids == self.config.image_token_id
                for pos in latent_pos + [seq_len]:
                    # ------------------------------------------------------------------
                    # 2.2.1 forward text and image tokens in [prev_idx, pos).
                    # These are non-latent tokens between latent tokens.
                    # ------------------------------------------------------------------

                    if pos > prev_idx:
                        s, e = prev_idx, pos
                        # ------------------------------------------------------------------
                        # 2.2.1.1 compute the length of image tail in this segment [s, e). Note that the image only exists in SFT Stage 2, since we insert image tokens before each latent segment.
                        # tail_len is the number of the continuous image tokens at the end of this segment
                        # ------------------------------------------------------------------
                        seg_mask = image_mask[b, s:e-2]
                        if seg_mask.numel() == 0:
                            tail_len = 0
                        else:
                            tail_len = seg_mask.to(torch.int32).flip(0).cumprod(0).sum().item()

                        if tail_len == 0:
                            cut = e  # no image in this segment
                        else:
                            cut = e - 2 - tail_len  # image in this segment
                        # ------------------------------------------------------------------
                        # 2.2.1.2 text sub-segment (with gradient; only allow Key up to cut to avoid seeing future image tail)
                        # ------------------------------------------------------------------
                        if cut > s:
                            def _run_text_chunk(q0: int, q1: int, k1: int, past_kv):
                                seg_embeds = seq_embeds[:, q0:q1, :]
                                seg_pos_ids = pos_ids_s[:, :, q0:q1]
                                if attn_mask_4d_full is not None:
                                    seg_att_m = attn_mask_s[:, :, q0:q1, :k1]
                                else:
                                    seg_att_m = attn_mask_s[:, :k1]
                                text_vp, text_ds = _slice_deepstack(b, q0, q1)
                                out = run_segment_forward(
                                    language_model=self.language_model,
                                    seg_embeds=seg_embeds,
                                    seg_pos_ids=seg_pos_ids,
                                    seg_att_m=seg_att_m,
                                    past_kv=past_kv,
                                    need_hidden=(teacher_hidden_states_for_alignment is not None and teacher_hidden_states_for_alignment[0].dim()==3),
                                    no_grad_mode=False,
                                    seg_visual_pos_masks=text_vp,
                                    seg_deepstack_visual_embeds=text_ds,
                                    **kwargs,
                                )
                                return out

                            out = _run_text_chunk(s, cut, cut, past_kv)
                            # ------------------------------------------------------------------
                            # 2.2.1.3 compute alignment loss of the observation tokens in text sub-segment (SFT Stage 2 student CoT))
                            # ------------------------------------------------------------------
                            if teacher_hidden_states_for_alignment is not None:  # align all layers
                                num_layers = len(out.hidden_states)
                                hidden_states_layers_b = [[] for _ in range(num_layers)]
                                teacher_align_poss = []
                                while align_ptr < len(align_pos) and align_pos[align_ptr] < cut:
                                    p = align_pos[align_ptr]
                                    offset = p - s
                                    for l in range(num_layers):
                                        vec = out.hidden_states[l][0, offset, :].flatten()
                                        hidden_states_layers_b[l].append(vec)
                                    teacher_align_poss.append(align_ptr)
                                    align_ptr += 1
                                if len(teacher_align_poss) > 0:
                                    student_hidden_states = [torch.stack(hid, dim=0) for hid in hidden_states_layers_b]
                                    student_hidden_states = torch.stack(student_hidden_states, dim=0)  # [num_layer, num_align_in_a_seg, dim]
                                    l_b = alignment_loss(
                                        teacher_hidden_states_for_alignment[b][:, teacher_align_poss, :],  # [num_layer, num_align_in_a_seg, dim]
                                        student_hidden_states,  # [num_layer, num_align_in_a_seg, dim]
                                    )
                                    align_losses_this_b += l_b

                            batch_last_hidden_state[b, s:cut, :] = out.last_hidden_state[0]
                            past_kv = out.past_key_values

                        # ------------------------------------------------------------------
                        # 2.2.1.4 image tail (without using 4D attention to save memory)
                        # ------------------------------------------------------------------
                        if cut < e:
                            img_embeds = seq_embeds[:, cut:e, :].detach()
                            img_pos_ids = pos_ids_s[:, :, cut:e]
                            img_vp, img_ds = _slice_deepstack(b, cut, e)
                            img_out = run_segment_forward(
                                language_model=self.language_model,
                                seg_embeds=img_embeds,
                                seg_pos_ids=img_pos_ids,
                                seg_att_m=None,
                                past_kv=past_kv,
                                need_hidden=False,
                                no_grad_mode=False,
                                seg_visual_pos_masks=img_vp,
                                seg_deepstack_visual_embeds=img_ds,
                                **kwargs,
                            )
                            batch_last_hidden_state[b, cut:e, :] = img_out.last_hidden_state[0]
                            past_kv = img_out.past_key_values

                            # Skip alignment points that fall within the image tail, but advance the pointer to ensure monotonicity
                            while align_ptr < len(align_pos) and align_pos[align_ptr] < e:
                                align_ptr += 1

                    # done
                    if pos == seq_len:
                        break

                    # ------------------------------------------------------------------
                    # 2.2.2 forward the latent token at position `pos`
                    # ------------------------------------------------------------------
                    if pos == 0:  # the first token is latent token, use predefined init embedding or zero
                        latent_embed = (
                            getattr(self, "latent_init_embedding", None)
                            .to(device, dtype)
                            .view(1, 1, -1)
                            if hasattr(self, "latent_init_embedding")
                            else torch.zeros(1, 1, hidden_dim, device=device, dtype=dtype)
                        )
                    else:  # set the latent token embedding as the previous token's hidden state
                        prev_hidden = batch_last_hidden_state[b, pos - 1, :].unsqueeze(0).unsqueeze(0).contiguous()
                        latent_embed = prev_hidden.clone() if self.training else prev_hidden.detach()
                        ce_patch_pos[b].append(pos)
                        ce_patch_vec[b].append(latent_embed[0, 0])

                    step_pos_ids = pos_ids_s[:, :, pos : pos + 1]

                    step_out = self.language_model(
                        input_ids=None,
                        inputs_embeds=latent_embed.detach(),
                        position_ids=step_pos_ids,
                        attention_mask=attention_mask[b : b + 1][:, : pos + 1],
                        past_key_values=past_kv,
                        cache_position=torch.tensor([pos], device=device),
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                        visual_pos_masks=None,
                        deepstack_visual_embeds=None,
                        return_dict=True,
                        **kwargs,
                    )

                    # ------------------------------------------------------------------
                    # 2.2.2.1 compute the alignment loss at this latent token position (SFT Stage 3 only)
                    # ------------------------------------------------------------------
                    if teacher_hidden_states_for_alignment is not None:
                        if align_ptr < len(align_pos) and align_pos[align_ptr] == pos:
                            if teacher_hidden_states_for_alignment[0].dim()==2:  # align only last hidden states (latent embeds)
                                align_losses_this_b += alignment_loss(
                                    teacher_hidden_states_for_alignment[b][align_ptr],  # [dim]
                                    latent_embed[0, 0]).unsqueeze(0)
                            elif teacher_hidden_states_for_alignment[0].dim()==3:  # align all layers
                                align_losses_this_b += alignment_loss(
                                    teacher_hidden_states_for_alignment[b][:, align_ptr, :],  # [dim]
                                    step_out.hidden_states)
                            align_ptr += 1

                    batch_last_hidden_state[b, pos, :] = step_out.last_hidden_state[0, 0]
                    hidden_states_to_return[b].append(step_out.hidden_states)
                    past_kv = step_out.past_key_values
                    prev_idx = pos + 1

                # ------------------------------------------------------------------
                # 2.3 Add the alignment loss of this sample to total alignment loss
                # ------------------------------------------------------------------
                past_key_values_batch[b] = past_kv
                if teacher_hidden_states_for_alignment is not None:
                    if align_losses_this_b > 0:
                        # normalize inside-tasks however you want
                        align_loss_b = align_losses_this_b / max(1, len(align_pos))
                    else:
                        align_loss_b = torch.zeros((), device=device, dtype=dtype)

                    if total_align_loss is None:
                        total_align_loss = align_loss_b
                    else:
                        total_align_loss += align_loss_b
            # ---------- end for-batch loop ----------
            if teacher_hidden_states_for_alignment is not None:
                total_align_loss = total_align_loss / batch_size  # or .sum()

            # ------------------------------------------------------------------
            # 3. Collect the latent token embeddings for the next CE loss forward stage
            # ------------------------------------------------------------------
            ce_patch_vec_tensors = []
            for b in range(batch_size):
                if len(ce_patch_vec[b]) > 0:
                    ce_patch_vec_tensors.append(torch.stack(ce_patch_vec[b], dim=0))  # (num_latents_b, H)
                else:
                    ce_patch_vec_tensors.append(torch.empty(0, hidden_dim, device=device, dtype=dtype))

            # ------------------------------------------------------------------
            # 4. Return
            # ------------------------------------------------------------------
            output = Qwen3VLModelOutputWithPast(
                alignment_loss=total_align_loss,
                last_hidden_state=batch_last_hidden_state,
                past_key_values=past_key_values_batch,
                hidden_states=hidden_states_to_return if output_hidden_states else None,
                attentions=None,
                rope_deltas=self.rope_deltas,
                ce_patch_pos=ce_patch_pos,                 # List[List[int]]
                ce_patch_vec=ce_patch_vec_tensors,          # List[Tensor(num_latents_b, H)]
            )
            return output if return_dict else output.to_tuple()
        # ---------- END latent_mode implementation ----------

        else:  # latent_mode == False
            if ce_patch_pos is not None and ce_patch_vec is not None:  # latent-ce_loss forward
                for b in range(len(ce_patch_pos)):
                    pos_list = ce_patch_pos[b]
                    if not pos_list:
                        continue
                    # (num_latents_b, H)
                    vecs = ce_patch_vec[b].to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds[b, torch.tensor(pos_list, device=inputs_embeds.device, dtype=torch.long), :] = vecs

            if "alignment" in kwargs.get('loss_type', {}):
                output_hidden_states = True

            outputs = self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attn_mask_4d_full if attn_mask_4d_full is not None else attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                return_dict=True,
                **kwargs,
            )

            if output_hidden_states:
                num_layers = len(outputs.hidden_states)
                hidden_states_to_return = []  # will return a list, each element is a sample in the batch with shape [num_layers, num_align or seq_len, dim]
                for b in range(batch_size):
                    hidden_states_b_tensor = []
                    for l in range(num_layers):
                        if alignment_poss[0]:
                            hidden_states_b_tensor.append(outputs.hidden_states[l][b, alignment_poss[b], :])
                        else:
                            hidden_states_b_tensor.append(outputs.hidden_states[l][b, :, :])
                    hidden_states_b_tensor = torch.stack(hidden_states_b_tensor, dim=0)  # [num_layers, num_align, dim]
                    hidden_states_to_return.append(hidden_states_b_tensor)

            total_align_loss = None
            if "alignment" in kwargs.get('loss_type', {}):
                total_align_loss = 0.
                all_student_hidden_states = torch.stack(outputs.hidden_states, dim=0)  # [num_layer, batch_size, seq_len, dim]
                for b in range(batch_size):
                    student_hidden_states = all_student_hidden_states[:, b, alignment_poss[b], :]
                    total_align_loss += alignment_loss(
                                            teacher_hidden_states_for_alignment[b],  # [num_layer, num_align_in_a_seg, dim]
                                            student_hidden_states,  # [num_layer, num_align_in_a_seg, dim]
                                        )
                total_align_loss /= batch_size
                output_hidden_states = False

            output = Qwen3VLModelOutputWithPast(
                alignment_loss=total_align_loss,
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states_to_return if output_hidden_states else None,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
            return output if return_dict else output.to_tuple()


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen3VL causal language model (or autoregressive) outputs.
    """
)
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    alignment_poss: Optional[List[List]] = None
    ce_patch_pos: Optional[List[List[int]]] = None
    ce_patch_vec: Optional[List[torch.Tensor]] = None
    latent_embeds: Optional[List[torch.Tensor]] = None
    mean_emphasize_acc: Optional[float] = None
    loss_dict: Optional[dict] = None  # Optional dictionary of losses when multiple objectives are computed in a single forward


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # student (latent mode), compatible with original Qwen3-VL
        attention_mask: Optional[torch.Tensor] = None,  # student (latent mode)
        attention_mask_4d: Optional[dict] = None,  # student (latent mode)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # student (latent mode)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,  # student (latent mode)
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        alignment_poss: Optional[List[List]] = None,
        latent_mode: Optional[bool] = False,
        teacher_hidden_states_for_alignment: Optional[List[List[torch.Tensor]]] = None,  # corresponding to the teacher hidden states of the observation tokens in SFT Stage 2, or the target hidden states of the latent tokens in SFT Stage 3
        ce_patch_pos: Optional[List[List[int]]] = None,
        ce_patch_vec: Optional[List[torch.Tensor]] = None,
        ce_emphasize_factor: Optional[float] = 1.0,
        ce_emphasize_poss: Optional[List[List[int]]] = None,
        loss_type: Optional[List[str]] = [],
        output_latent_embeds: bool = False,
        compute_emphasize_acc: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if not latent_mode and 'alignment' in loss_type:
            kwargs['loss_type'] = loss_type

        # Forward through the MLLM.
        # Note that the latent forward is in it.
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attention_mask_4d=attention_mask_4d,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            latent_mode=latent_mode,
            alignment_poss=alignment_poss,
            teacher_hidden_states_for_alignment=teacher_hidden_states_for_alignment,
            ce_patch_pos=ce_patch_pos,
            ce_patch_vec=ce_patch_vec,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        mean_emphasize_acc = None
        logits = None
        loss = 0.
        loss_dict = {}
        if "ce" in loss_type:
            # Optional: apply per-token weighting on ce_emphasize_poss
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            use_weight = (
                ce_emphasize_poss is not None and isinstance(ce_emphasize_poss, (list, tuple)) and len(ce_emphasize_poss) > 0 and
                isinstance(ce_emphasize_factor, (int, float)) and ce_emphasize_factor is not None and float(ce_emphasize_factor) != 1.0
            )
            if use_weight:  # Use weighted CE loss, will scale up the loss on selected positions specified in ce_emphasize_poss
                # Compute token-wise CE with ignore_index and apply weights on selected positions
                B, S, V = logits.shape
                logits_flat = logits.view(-1, V)
                labels = nn.functional.pad(labels, (0, 1), value=-100)
                shift_labels = labels[..., 1:].contiguous()
                shift_labels_flat = shift_labels.view(-1)
                ce_flat = F.cross_entropy(logits_flat, shift_labels_flat, reduction='none', ignore_index=-100)
                ce = ce_flat.view(B, S)

                # Build weights mask
                weight = torch.ones_like(ce)
                try:
                    for b, poss in enumerate(ce_emphasize_poss):
                        if not poss:
                            continue
                        weight[b, torch.tensor(poss)-1] = float(ce_emphasize_factor)
                except Exception:
                    # Fallback to unweighted if ce_emphasize_poss malformed
                    weight = torch.ones_like(ce)

                valid = (shift_labels != -100).float()
                num_valid = (weight * valid).sum().clamp_min(1.0)
                loss = (ce * weight * valid).sum() / num_valid
                loss_dict['ce'] = loss
            else:
                # Fallback to default loss function
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)
                loss_dict['ce'] = loss

            has_obs_cnt = 0

        if compute_emphasize_acc:  # Compute token prediction accuracy on the emphasize positions
            with torch.no_grad():
                mean_emphasize_acc = 0
                for b in range(labels.shape[0]):
                    if not ce_emphasize_poss[b]:
                        continue
                    has_obs_cnt += 1
                    poss = torch.tensor(ce_emphasize_poss[b], device=labels.device, dtype=torch.long)
                    preds = logits[b].argmax(dim=-1)[poss-1][:]
                    emphasize_labels = labels[b][poss][:].to(preds.device)
                    correct = (preds == emphasize_labels).float()
                    emphasize_acc = correct.sum() / max(1, poss.shape[0])
                    mean_emphasize_acc += emphasize_acc.item()
                if mean_emphasize_acc>0:
                    mean_emphasize_acc /= has_obs_cnt

        if 'alignment' in loss_type:
            loss_dict['alignment'] = outputs.alignment_loss

        latent_embeds = None
        if output_latent_embeds:
            assert latent_mode, "output_latent_embeds requires latent_mode"
            latent_embeds = []
            for b, alignment_poss_b in enumerate(alignment_poss):
                latent_embeds.append(outputs.last_hidden_state[b][alignment_poss_b, :].detach())

        hidden_states_to_return = None
        if output_hidden_states and outputs.hidden_states is not None:
            if isinstance(outputs.hidden_states[0], list):  # return from latent_mode
                hidden_states_to_return = []
                for hidden_state_b_list in outputs.hidden_states:
                    hidden_state_b_tensor = []
                    for hidden_state_all_layers_p_tuple in hidden_state_b_list:
                        hidden_state_all_layers_p_tensor = torch.cat(hidden_state_all_layers_p_tuple, dim=0)  # (num_layers, 1, hidden_dim)
                        hidden_state_b_tensor.append(hidden_state_all_layers_p_tensor)
                    hidden_state_b_tensor = torch.cat(hidden_state_b_tensor, dim=1)  # (num_layers, num_latents_b, hidden_dim)
                    hidden_states_to_return.append(hidden_state_b_tensor)
            elif isinstance(outputs.hidden_states[0], torch.Tensor):  # return from non-latent_mode
                hidden_states_to_return = outputs.hidden_states  # List[Tensor], each (B, S, H), S is total seq len (alignment poss is not set) or num_align (alignment poss is set)

        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states_to_return,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            alignment_poss=alignment_poss,
            ce_patch_pos=outputs.ce_patch_pos if hasattr(outputs, "ce_patch_pos") else None,
            ce_patch_vec=outputs.ce_patch_vec if hasattr(outputs, "ce_patch_vec") else None,
            latent_embeds=latent_embeds,
            mean_emphasize_acc=mean_emphasize_acc,
            loss_dict=loss_dict if len(loss_dict) > 0 else None,
        )
        return output if return_dict else output.to_tuple()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
]
