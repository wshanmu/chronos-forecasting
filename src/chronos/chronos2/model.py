# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import copy
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from chronos.chronos_bolt import InstanceNorm, Patch

from .config import Chronos2CoreConfig, Chronos2ForecastingConfig
from .layers import (
    MHA,
    MLP,
    AttentionOutput,
    Chronos2LayerNorm,
    FeedForward,
    GroupSelfAttention,
    ResidualBlock,
    TimeSelfAttention,
)


@dataclass
class Chronos2EncoderBlockOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    time_self_attn_weights: torch.Tensor | None = None
    group_self_attn_weights: torch.Tensor | None = None


class Chronos2EncoderBlock(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        assert not config.is_decoder

        self.layer = nn.ModuleList()
        self.layer.append(TimeSelfAttention(config))
        self.layer.append(GroupSelfAttention(config))
        self.layer.append(FeedForward(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Chronos2EncoderBlockOutput:
        # apply time attention
        time_self_attn_outputs: AttentionOutput = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = time_self_attn_outputs[0]

        # apply group attention
        group_self_attn_outputs: AttentionOutput = self.layer[1](
            hidden_states, attention_mask=group_time_mask, output_attentions=output_attentions
        )
        hidden_states = group_self_attn_outputs[0]

        # apply feed forward layer
        hidden_states = self.layer[2](hidden_states)

        return Chronos2EncoderBlockOutput(
            hidden_states=hidden_states,
            time_self_attn_weights=time_self_attn_outputs.attn_weights,
            group_self_attn_weights=group_self_attn_outputs.attn_weights,
        )


@dataclass
class Chronos2EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    all_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    all_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    all_hidden_states: tuple[torch.Tensor, ...] | None = None


class Chronos2Encoder(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        assert not config.is_decoder

        self.block = nn.ModuleList([Chronos2EncoderBlock(config) for i in range(config.num_layers)])
        self.final_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        assert attention_mask.ndim == 2, "attention_mask must have shape (batch, seq_len)"

        # Add new dims for attention heads and q_len
        attention_mask = attention_mask[:, None, None, :]

        # Invert binary mask to float mask which can be added to attention scores
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    @staticmethod
    def _construct_and_invert_group_time_mask(
        group_ids: torch.Tensor, attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        # construct group_mask (batch, batch) from group ids
        # a cell is True if both row and col had the same group id
        group_mask = group_ids[:, None] == group_ids[None, :]
        # outer product of group_mask and attention_mask (time_mask)
        # group_time_mask combines group and time masks to ensure that attention only uses
        # tokens from the same group which are also not masked in time
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)

        if torch.is_floating_point(group_time_mask):
            # this ensures that mixed precision training does not overflow
            floating_type = group_time_mask.dtype

        # reshape mask to shape of attention scores
        group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
        group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min

        return group_time_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Chronos2EncoderOutput:
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # make the time attention mask broadcastable to attention scores (batch, n_heads, q_len, kv_len) and invert
        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, inputs_embeds.dtype)

        # construct group time mask
        group_time_mask = self._construct_and_invert_group_time_mask(group_ids, attention_mask, inputs_embeds.dtype)

        all_time_self_attentions: tuple[torch.Tensor, ...] = ()
        all_group_self_attentions: tuple[torch.Tensor, ...] = ()
        all_hidden_states: tuple[torch.Tensor, ...] = () if output_hidden_states else None
        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module) in enumerate(self.block):
            layer_outputs: Chronos2EncoderBlockOutput = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                assert layer_outputs.time_self_attn_weights is not None
                assert layer_outputs.group_self_attn_weights is not None

                all_time_self_attentions = (*all_time_self_attentions, layer_outputs.time_self_attn_weights)
                all_group_self_attentions = (*all_group_self_attentions, layer_outputs.group_self_attn_weights)

            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return Chronos2EncoderOutput(
            last_hidden_state=hidden_states,
            all_time_self_attn_weights=all_time_self_attentions,
            all_group_self_attn_weights=all_group_self_attentions,
            all_hidden_states=all_hidden_states,
        )


@dataclass
class Chronos2Output(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    enc_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    enc_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None

@dataclass
class Chronos2ClassificationOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None


class Chronos2Model(PreTrainedModel):
    config_class = Chronos2CoreConfig  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True

    def __init__(self, config: Chronos2CoreConfig):
        assert hasattr(config, "chronos_config"), "Not a valid Chronos config"

        super().__init__(config)
        self.config: Chronos2CoreConfig
        self.model_dim = config.d_model

        config.chronos_config["time_encoding_scale"] = config.chronos_config.get(
            "time_encoding_scale", config.chronos_config["context_length"]
        )
        self.chronos_config = Chronos2ForecastingConfig(**config.chronos_config)

        assert self.chronos_config.input_patch_size == self.chronos_config.output_patch_size, (
            "input_patch_size and output_patch_size sizes must be equal, "
            f"but found {self.chronos_config.input_patch_size} and {self.chronos_config.output_patch_size}"
        )

        # Only [PAD] token (and [REG] token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.chronos_config.input_patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size, patch_stride=self.chronos_config.input_patch_stride
        )

        patch_size_stats = self.chronos_config.__dict__.get("patch_size_stats", 32)
        patch_stride_stats = self.chronos_config.__dict__.get("patch_stride_stats", 32)
        self.patch_for_stats = Patch(
            patch_size=patch_size_stats, patch_stride=patch_stride_stats
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = Chronos2Encoder(encoder_config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.output_patch_size,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, Chronos2LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, MLP):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MHA):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            kv_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * kv_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * kv_proj_dim) ** -0.5))
        elif isinstance(module, (Chronos2Model)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def _validate_input(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        num_output_patches: int,
        future_target: torch.Tensor | None,
        future_target_mask: torch.Tensor | None,
    ):
        output_patch_size = self.chronos_config.output_patch_size
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch_size, context_length), found: {tuple(context.shape)}")
        if context_mask is not None and context_mask.shape != context.shape:
            raise ValueError(f"mask must have shape {tuple(context.shape)}, found: {tuple(context_mask.shape)}")
        if future_covariates is not None:
            if future_covariates.shape[0] != context.shape[0] or future_covariates.ndim != 2:
                raise ValueError(
                    f"future_covariates must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_covariates.shape)}"
                )
            if future_covariates.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_covariates, "
                    f"found: {future_covariates.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
            if future_target is not None and future_target.shape != future_covariates.shape:
                raise ValueError(
                    f"future_target must have the same shape as future_covariates, found: {tuple(future_target.shape)} and {tuple(future_covariates.shape)}"
                )
        if future_covariates_mask is not None:
            if future_covariates is None:
                raise ValueError("future_covariates must be provided if future_covariates_mask is provided")
            if future_covariates_mask.shape != future_covariates.shape:
                raise ValueError(
                    f"future_covariates_mask must have the same shape as future_covariates, "
                    f"found: {tuple(future_covariates_mask.shape)} and {tuple(future_covariates.shape)}"
                )
        if group_ids is not None and group_ids.shape != (context.shape[0],):
            raise ValueError(f"group_ids must have shape (batch_size,), found: {tuple(group_ids.shape)}")
        if future_target is not None:
            if future_target.shape[0] != context.shape[0] or future_target.ndim != 2:
                raise ValueError(
                    f"future_target must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_target.shape)}"
                )
            if future_target.shape[-1] > output_patch_size * num_output_patches:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_target, "
                    f"found: {future_target.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
        if future_target_mask is not None:
            if future_target is None:
                raise ValueError("future_target must be provided if future_target_mask is provided")
            if future_target_mask.shape != future_target.shape:
                raise ValueError(
                    f"future_target_mask must have the same shape as future_target, found: {tuple(future_target_mask.shape)} and {tuple(future_target.shape)}"
                )

    def _prepare_patched_context(
        self, 
        context: torch.Tensor, 
        context_mask: torch.Tensor | None = None,
        return_patch_stats: bool = False  # Default to False for backward compatibility
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, context_length = context.shape
        # truncate context if it's longer than model's context length
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

        # 1. Capture RAW context for Absolute Sample Statistics ONLY if requested
        raw_context = context.clone() if return_patch_stats else None

        # scaling
        context, loc_scale = self.instance_norm(context)

        # scaling is done in 32-bit precision, then the context is moved to model's dtype
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # context time encoding: every observation is assigned a sequential time index,
        # scaled by model's context length = [-C, -(C-1), ..., -1] / context_length
        final_context_length = num_context_patches * self.chronos_config.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=self.device, dtype=torch.float32)
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.chronos_config.input_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        # concat time encoding, context and mask along the last (feature) dim
        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)

        # 2. Conditional Return Logic
        if return_patch_stats:
            # Calculate patch-wise statistics on unscaled data (Stat Augmentation)
            raw_patched = self.patch_for_stats(raw_context)
            p_mean = torch.nanmean(raw_patched, dim=-1, keepdim=True)
            p_std = torch.std(raw_patched, dim=-1, keepdim=True) # Replaced numpy with torch for speed
            p_min = torch.amin(torch.nan_to_num(raw_patched, nan=float('inf')), dim=-1, keepdim=True)
            p_max = torch.amax(torch.nan_to_num(raw_patched, nan=float('-inf')), dim=-1, keepdim=True)
            
            patch_stats = torch.cat([p_mean, p_std, p_min, p_max], dim=-1).to(self.dtype) # Shape: (batch_size, num_patches, 4)
            
            return patched_context, attention_mask, loc_scale, patch_stats
        
        # Default behavior: return only the original 3 elements
        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = cast(torch.Tensor, future_covariates)
            future_covariates = future_covariates.to(self.dtype)

            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)

            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)

            if torch.isnan(future_covariates).any():
                raise ValueError(
                    "future_covariates contains NaN values at indices not masked by future_covariates_mask. "
                    "Input the correct future_covariates_mask or omit it to automatically infer the mask based on NaN values."
                )

            # add padding if the length of future_covariates is not an integer multiple of output_patch_size
            if num_output_patches * output_patch_size > future_covariates.shape[-1]:
                padding_shape = (
                    *future_covariates.shape[:-1],
                    num_output_patches * output_patch_size - future_covariates.shape[-1],
                )
                future_covariates = torch.cat(
                    [future_covariates, torch.zeros(padding_shape).to(future_covariates)], dim=-1
                )
                future_covariates_mask = torch.cat(
                    [future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)], dim=-1
                )

            patched_future_covariates = rearrange(
                future_covariates, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )

        # future time encoding: every future timestep is assigned a sequential time index,
        # scaled by model's context length = [0, 1, ..., h-1] / context_length
        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(start=0, end=final_future_length, device=self.device, dtype=torch.float32)
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_output_patches,
                p=output_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_future = torch.cat(
            [future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1
        )

        return patched_future, patched_future_covariates_mask

    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size
        assert quantile_preds.shape[0] == batch_size and quantile_preds.shape[-1] >= future_target.shape[-1]

        # normalize target and mask
        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1)
        future_target = future_target.to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        # pad target and target_mask if they are shorter than model's prediction
        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (*future_target.shape[:-1], quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat([future_target, torch.zeros(padding_shape).to(future_target)], dim=-1)
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(padding_shape).to(future_target_mask)], dim=-1
            )

        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles)
        )
        inv_future_covariate_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size,
            n=num_output_patches,
            p=output_patch_size,
        )
        # the first components masks any missing targets and the second component masks known future values
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        loss = quantile_loss * loss_mask
        # mean over prediction horizon, sum over quantile levels and mean over batch
        loss = loss.mean(dim=-1).sum(dim=-1).mean()

        return loss

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        num_context_patches = attention_mask.shape[-1]

        # get input embeddings of shape (batch, num_context_patches, d_model)
        input_embeds: torch.Tensor = self.input_patch_embedding(patched_context)
        # append [REG] special token embedding, if needed
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)

        # get future embeddings of shape (batch, num_output_patches, d_model)
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # concatenate context and future embeddings and masks
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            # by default, each time series is treated independently, i.e., no mixing across the batch
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs: Chronos2EncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2Output:
        """Forward pass of the Chronos2 model.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        group_ids : torch.Tensor | None, optional
            Group IDs of shape (batch_size,) indicating which times series in the batch form a group.
            A group indicates a task, for example, for a batch of size 6:
            - if groups_ids = [0, 1, 2, 3, 4, 5], each time series is treated independently.
            - if groups_ids = [0, 0, 1, 1, 1, 2], information is mixed across the first two time series (id=0),
                the next three time series (id=1) and the last time series is treated separately. Information is
                NOT shared among time series from different groups.
            The ordering and specific values of group_ids are not important, all time series with the same group
            ID form a group.
        future_covariates
            Tensor of shape (batch_size, future_length) containing future covariates. Note that the size of
            tensor along the first axis is equal to the batch_size. This means that future values (which may be NaNs)
            must be provided for each time series in the batch. For any time series that need to be forecasted, the
            future_covariates can be set to NaNs, if ``future_covariates_mask`` is omitted or to an arbitrary dummy
            value when ``future_covariates_mask`` is provided. ``future_covariates`` can be used with ``group_ids``
            to construct heterogenous forecasting tasks in a single batch. For example:
            - future_covariates = [[nan, ...], [nan, ...], [v1, ...], [v2, ...], [nan, ...], [nan, ...]]
            - groups_ids = [0, 0, 1, 1, 1, 2]
            - future_covariates_mask = None
            contains 3 types of forecasting tasks:
            - [0, 0]: The first task, both future_covariates are missing, which implies that the two time series need to
                be forecasted jointly, i.e., multivariate forecasting.
            - [1, 1, 1]: In the next task, the first two future_covariates are available and the last one is missing
                ([v1, ...], [v2, ...], [nan, ...]), where [v1, ...] and [v1, ...] denote an arbitrary sequence of values.
                This indicates that the first two time series are known covariates and the third one needs to be forecasted
                by the model.
            - [2]: The last task has a single time series in the group which needs to be forecasted independently.
            There is no theoretical limit on the number of time series in a group, i.e., the number of targets and known
            covariates in a task. The above setup subsumes tasks with past-only covariates as the model's prediction for
            those time series can simply be ignored downstream.
        future_covariates_mask
            Binary mask tensor of same shape as future_covariates indicating which future values are known
            If omitted, future_covariates_mask is automatically constructed based on future_covariates with
            all non-NaN values treated as known future values.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
            When ``future_covariates`` and/or ``future_target`` are provided, num_output_patches should be large enough to accommodate
            their lengths, i.e., num_output_patches * output_patch_size >= future_length
        future_target
            Target tensor of shape (batch_size, future_length) used during training. If ``future_covariates`` are provided, both
            target and future_covariates must have the same shape.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        Chronos2Output containing:
        - loss: Training loss, if `future_target` is provided
        - quantile_preds: Quantile predictions of shape (batch_size, num_quantiles, num_output_patches * output_patch_size).
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it was a
            known future covariate.
        - enc_time_self_attn_weights: Time self attention weights, if output_attentions=True
        - enc_group_self_attn_weights: Group self attention weights, if output_attentions=True
        """
        batch_size = context.shape[0]
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )
        hidden_states: torch.Tensor = encoder_outputs[0]
        assert hidden_states.shape == (batch_size, num_context_patches + 1 + num_output_patches, self.model_dim)

        # slice the last num_output_patches hidden states to be input into the output_patch_embedding
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        loss = (
            self._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
            if future_target is not None
            else None
        )

        # Unscale predictions
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )

        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )

class Chronos2ModelClassification(Chronos2Model):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_classes = config.chronos_config["n_classes"]
        self.output_all_hidden_states = config.chronos_config["output_all_hidden_states"]
        self.n_channels = config.chronos_config["n_channels"]
        
        # Remove the forecasting head to save memory/VRAM
        del self.output_patch_embedding
        self.context_length = config.chronos_config["context_length"]
        if self.output_all_hidden_states:
            self.final_dim = (self.model_dim * 12 + 2) * self.n_channels  
        else:
            self.final_dim = (self.model_dim + 2 + self.context_length//32 * 4) * self.n_channels # (model_dim + instance_norm feature + patch_stats) * num_channel
        self.classification_head = nn.Sequential(
            nn.Linear(self.final_dim, 64),
            nn.ReLU(),
            # nn.Linear(512, 64),
            # nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.supcon_mode = False
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.geometry_projector = nn.Sequential(
            nn.Linear(20, self.model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.model_dim // 2, self.model_dim)
        )

        # Call the initialization
        self.post_init_custom()

    def post_init_custom(self):
        for head in [self.classification_head, self.projection_head, self.geometry_projector]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Critically, initialize the weights and bias of the final Linear layer to zero
        nn.init.zeros_(self.geometry_projector[-1].weight)
        if self.geometry_projector[-1].bias is not None:
            nn.init.zeros_(self.geometry_projector[-1].bias)
    
    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        bin_centers: torch.Tensor | None = None,
    ):
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale, patch_stats = self._prepare_patched_context(
            context=context, context_mask=context_mask, return_patch_stats=True
        )
        num_context_patches = attention_mask.shape[-1]

        # get input embeddings of shape (batch, num_context_patches, d_model)
        input_embeds: torch.Tensor = self.input_patch_embedding(patched_context)

        if bin_centers is not None:
            import math
            # 1. Normalization
            norm_centers = (bin_centers.float() - 10.0) / (54.0 - 10.0)
            norm_centers = torch.clamp(norm_centers, 0.0, 1.0)
                
            # 2. Spectral Expansion
            k_exp = torch.pow(2.0, torch.arange(10, device=bin_centers.device, dtype=torch.float32))

            # PerceptAlign uses pi * p * 2^k
            angle = math.pi * norm_centers.unsqueeze(1) * k_exp.unsqueeze(0)
            geom_features = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) # [BS, 20]
            
            # 3. Geometry Projector
            geom_embeds = self.geometry_projector(geom_features)
            
            # 5. Integration
            input_embeds = input_embeds + geom_embeds.unsqueeze(1)

        # append [REG] special token embedding, if needed
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )

        if group_ids is None:
            # by default, each time series is treated independently, i.e., no mixing across the batch
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs: Chronos2EncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        return encoder_outputs, loc_scale, num_context_patches, patch_stats

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        labels=None,
        bin_centers=None,
    ) -> Chronos2Output:
        """Forward pass of the Chronos2 model.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        group_ids : torch.Tensor | None, optional
            Group IDs of shape (batch_size,) indicating which times series in the batch form a group.
            A group indicates a task, for example, for a batch of size 6:
            - if groups_ids = [0, 1, 2, 3, 4, 5], each time series is treated independently.
            - if groups_ids = [0, 0, 1, 1, 1, 2], information is mixed across the first two time series (id=0),
                the next three time series (id=1) and the last time series is treated separately. Information is
                NOT shared among time series from different groups.
            The ordering and specific values of group_ids are not important, all time series with the same group
            ID form a group.
        future_covariates
            Tensor of shape (batch_size, future_length) containing future covariates. Note that the size of
            tensor along the first axis is equal to the batch_size. This means that future values (which may be NaNs)
            must be provided for each time series in the batch. For any time series that need to be forecasted, the
            future_covariates can be set to NaNs, if ``future_covariates_mask`` is omitted or to an arbitrary dummy
            value when ``future_covariates_mask`` is provided. ``future_covariates`` can be used with ``group_ids``
            to construct heterogenous forecasting tasks in a single batch. For example:
            - future_covariates = [[nan, ...], [nan, ...], [v1, ...], [v2, ...], [nan, ...], [nan, ...]]
            - groups_ids = [0, 0, 1, 1, 1, 2]
            - future_covariates_mask = None
            contains 3 types of forecasting tasks:
            - [0, 0]: The first task, both future_covariates are missing, which implies that the two time series need to
                be forecasted jointly, i.e., multivariate forecasting.
            - [1, 1, 1]: In the next task, the first two future_covariates are available and the last one is missing
                ([v1, ...], [v2, ...], [nan, ...]), where [v1, ...] and [v1, ...] denote an arbitrary sequence of values.
                This indicates that the first two time series are known covariates and the third one needs to be forecasted
                by the model.
            - [2]: The last task has a single time series in the group which needs to be forecasted independently.
            There is no theoretical limit on the number of time series in a group, i.e., the number of targets and known
            covariates in a task. The above setup subsumes tasks with past-only covariates as the model's prediction for
            those time series can simply be ignored downstream.
        future_covariates_mask
            Binary mask tensor of same shape as future_covariates indicating which future values are known
            If omitted, future_covariates_mask is automatically constructed based on future_covariates with
            all non-NaN values treated as known future values.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
            When ``future_covariates`` and/or ``future_target`` are provided, num_output_patches should be large enough to accommodate
            their lengths, i.e., num_output_patches * output_patch_size >= future_length
        future_target
            Target tensor of shape (batch_size, future_length) used during training. If ``future_covariates`` are provided, both
            target and future_covariates must have the same shape.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        Chronos2Output containing:
        - loss: Training loss, if `future_target` is provided
        - quantile_preds: Quantile predictions of shape (batch_size, num_quantiles, num_output_patches * output_patch_size).
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it was a
            known future covariate.
        - enc_time_self_attn_weights: Time self attention weights, if output_attentions=True
        - enc_group_self_attn_weights: Group self attention weights, if output_attentions=True
        """
        
        # context: [bs*num_vars, context_length]
        # group_ids: [bs*num_vars,] 
        
        batch_size = context.shape[0]
        encoder_outputs, loc_scale, num_context_patches, patch_stats = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
            bin_centers=bin_centers,
        )
        # loc_scale is the scaling parameters (bs, 2);
        loc_scale_tensor = torch.cat(loc_scale, dim=-1)
        # patch_stats is the statistics for each patch, Shape: (batch_size, num_patches, 4)]
        patch_stats = patch_stats.view(batch_size, -1)

        hidden_states: torch.Tensor = encoder_outputs[0]
        assert hidden_states.shape == (batch_size, num_context_patches + 1, self.model_dim)

        all_hidden_states = encoder_outputs.all_hidden_states # 12*[B*N, num_patches, d_model]
        pooled = [torch.mean(layer, dim=1) for layer in all_hidden_states]
        if self.output_all_hidden_states:
            # then concatenate num_var such vectors for multivariate classification -> based on group_ids shape (BS,)
            # Mean pool each layer across the sequence dimension
            combined_features = torch.cat(pooled, dim=-1) # Shape: [batch*num_vars, d_model * 12]
        else:
            combined_features = pooled[-1]

        combined_features_with_statistics = torch.cat([combined_features, loc_scale_tensor, patch_stats], dim=1) # Shape: [batch*num_vars, d_model * 12 + 2]
        
        ## If we want to mean pool over I/Q per link:
        # combined_features_with_statistics = combined_features_with_statistics.view(batch_size // 8, 2, 4, -1) # 4 is num_vars TODO: make this dynamic based on group_ids
        # combined_features_with_statistics = (torch.mean(combined_features_with_statistics, dim=1)).view(batch_size//8, -1)
        
        # Determine the number of variables (channels) per sample in the batch dynamically
        # context shape is [BS * num_vars, context_length]
        # We need to reshape back to [BS, num_vars * features]
        # Since group_ids is [BS * num_vars], its unique values should give us BS.
        actual_bs = len(torch.unique(group_ids)) if group_ids is not None else batch_size // self.n_channels
        combined_features_with_statistics = combined_features_with_statistics.view(actual_bs, -1)
        # combined_features_with_statistics = combined_features_with_statistics.view(batch_size // self.n_channels, -1) # 4 is num_vars TODO: make this dynamic based on group_ids
        
        if getattr(self, "supcon_mode", False):
            # 3. Projection Head
            z = self.projection_head(combined_features_with_statistics)
            z = F.normalize(z, dim=1)
            
            # Split into view1 and view2
            B = z.shape[0] // 2
            z1, z2 = z[:B], z[B:]
            
            # Combine -> [B, 2, proj_dim]
            features = torch.stack([z1, z2], dim=1)
            
            loss = None
            if labels is not None:
                simclr_temperature = self.chronos_config.__dict__.get("simclr_temperature", 0.2)
                # supcon_loss_fct = SupConLoss(temperature=simclr_temperature)
                # loss = supcon_loss_fct(features, labels)
                # Experimenting with SimCLR style (unsupervised)
                simclr_loss_fct = SimCLRLoss(temperature=simclr_temperature)
                loss = simclr_loss_fct(features)
            # Generic return satisfying HF Trainer
            return Chronos2ClassificationOutput(
                loss=loss,
                logits=z1,
                embeddings=combined_features_with_statistics,
            )
            
        # 3. Classification Head
        logits = self.classification_head(combined_features_with_statistics) # [BS, num_classes]
        
        # 4. Loss Calculation (Required for HF Trainer)
        loss = None
        if labels is not None:
            train_loss_type = self.chronos_config.__dict__.get("train_loss", "cross_entropy")
            if train_loss_type == "cross_entropy":
                loss_fct = nn.CrossEntropyLoss()
            elif train_loss_type == "focal_loss":
                loss_fct = BinaryFocalLoss(alpha=0.25, gamma=2.0)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels.long())
        return Chronos2ClassificationOutput(
            loss=loss,
            logits=logits,
            embeddings=combined_features_with_statistics,
        )
        
import torch.nn.functional as F
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: model logits (before sigmoid)
        # targets: ground truth (0 or 1)
        
        # Calculate standard binary cross entropy
        bce_loss_fucntion = nn.CrossEntropyLoss()
        bce_loss = bce_loss_fucntion(inputs, targets)
        
        # Get the probability of the true class
        pt = torch.exp(-bce_loss) 
        
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask).repeat(anchor_count, contrast_count),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask.repeat(anchor_count, contrast_count) * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (torch.max(mask.sum(1), torch.ones_like(mask.sum(1))) + 1e-9)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: hidden vector of shape [batch_size, n_views, dim].
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Reshape to [batch_size * n_views, dim]
        # features are assumed to be already L2 normalized based on your snippet
        features = torch.cat(torch.unbind(features, dim=1), dim=0) 

        # Compute similarity matrix (2B x 2B)
        logits = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability: subtract the max logit in each row
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Create mask to exclude self-contrast (diagonal)
        # We need to find the "positive" for each anchor.
        # If views are [v1_1, v1_2, ... v1_B, v2_1, v2_2, ... v2_B]
        # The positive for v1_i is v2_i (index i + B)
        # The positive for v2_i is v1_i (index i - B)
        mask = torch.eye(batch_size * n_views, dtype=torch.bool).to(device)
        logits = logits.masked_fill(mask, -1e9) # Effectively zero out the diagonal in Exp

        # Targets: The index of the corresponding view
        # Example for 2 views: [B, B+1, ..., 2B-1, 0, 1, ..., B-1]
        targets = torch.arange(batch_size * n_views).to(device)
        targets = (targets + batch_size) % (batch_size * n_views)

        # Cross Entropy Loss
        loss = F.cross_entropy(logits, targets)

        return loss