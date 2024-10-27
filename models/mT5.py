import math, copy, random
from typing import *

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint

from transformers.utils import logging

from transformers import MT5ForConditionalGeneration
from transformers.models.mt5.modeling_mt5 import (
    MT5Block,
    MT5Stack,
    MT5LayerNorm,
    MT5PreTrainedModel,
    MT5LayerSelfAttention,
    MT5LayerCrossAttention,
    MT5LayerFF,
)
from transformers.models.mt5.configuration_mt5 import MT5Config

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.generation.beam_search import BeamHypotheses, BeamSearchScorer, ConstrainedBeamSearchScorer

from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    dist,
    inspect,
    is_deepspeed_zero3_enabled,
    warnings,
)
from math import ceil


from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)


def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device."
            " These attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]

    return dict(zip(devices, layers_list))



class Adapter(nn.Module): ### transformers/src/transformers/models/xmod/modeling_xmod.py XmodAdapter
    def __init__(self, config, adapter_reduction_factor, adapter_layer_norm):
        super().__init__()
        self.config = config
        self.bottleneck_size = config.hidden_size // adapter_reduction_factor
        self.dense1 = nn.Linear(config.hidden_size, self.bottleneck_size)
        self.dense2 = nn.Linear(self.bottleneck_size, config.hidden_size)
        if isinstance(config.adapter_hidden_act, str):
            self.adapter_act_fn = ACT2FN[config.adapter_hidden_act]
        else:
            self.adapter_act_fn = config.adapter_hidden_act

        self.adapter_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon) if adapter_layer_norm else None
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.adapter_layer_norm is not None:
            hidden_states = self.adapter_layer_norm(hidden_states)

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.adapter_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MT5BlockWithAdapter(MT5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(MT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MT5LayerCrossAttention(config))

        self.layer.append(MT5LayerFF(config))
        self.Adapter_dict = nn.ModuleDict({})

    def adapter_forward(
        self,
        adapter_modules: nn.ModuleDict,
        adapter_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        adapter_label_dict: dict,
    ) -> torch.Tensor:
        # adapter_ids, lang_lengths = torch.unique_consecutive(adapter_ids, return_counts=True)

        residual_states = hidden_states
        adapter_label = adapter_label_dict[adapter_ids[0].item()]
        hidden_states = adapter_modules[adapter_label](hidden_states)
        """
        split_hidden_states = torch.split(hidden_states, lang_lengths.tolist(), 0)
        adapter_wise_outputs = []
        for adapter_id, split_hidden_state in zip(adapter_ids, split_hidden_states):
            adapter_label = adapter_label_dict[adapter_id.item()]
            adapter_wise_outputs.append(adapter_modules[adapter_label](split_hidden_state))
        
        hidden_states = torch.cat(adapter_wise_outputs, 0)
        """
        hidden_states = hidden_states + residual_states
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        lang_ids=None,
        task_ids=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # Apply language-specific adapters #######################################################
        if "lang" in self.Adapter_dict:
            hidden_states = self.adapter_forward(self.Adapter_dict["lang"], lang_ids, hidden_states, adapter_label_dict=self.config.lang_dict)

        if "task" in self.Adapter_dict:
            hidden_states = self.adapter_forward(self.Adapter_dict["task"], task_ids, hidden_states, adapter_label_dict=self.config.task_dict)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class MT5StackwithAdapter(MT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [MT5BlockWithAdapter(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        lang_ids=None,
        task_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    lang_ids=lang_ids,
                    task_ids=task_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class LangAgnosticEncoder(MT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [MT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        lang_ids=None,
        task_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)


        # Remove language identity information here

        assert self.removal_type is not None, "Please specify the removal type"
        random_number = random.random()
        if random_number < self.remove_prob:
            if self.removal_type == "mean_zero":
                if self.is_synthesize:
                    # During inference, we feed english mr as an input - so we have to subtract english mean
                    lang_mean = self.lang_means[torch.zeros_like(lang_ids)]
                else:
                    lang_mean = self.lang_means[lang_ids]
                hidden_states = hidden_states - lang_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            elif self.removal_type == "mean_eng":
                # Shift to English mean, if not English && not pad
                if 0 not in lang_ids:
                    eng_mean = self.lang_means[torch.zeros_like(lang_ids)]
                    lang_mean = self.lang_means[lang_ids]
                    hidden_states = hidden_states - lang_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape) + eng_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            elif self.removal_type == "proj_remove":
                attention_mask_reshaped = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
                sentence_rep = torch.sum(hidden_states * attention_mask_reshaped, dim=1) / torch.clamp(attention_mask_reshaped.sum(1), min=1e-9) # We ignore the padded tokens (B, D)
                sentence_rep_l2norm = torch.norm(sentence_rep, dim=-1, p=2, keepdim=True) #(B, 1)
                if self.is_synthesize:
                    # During inference, we feed english mr as an input - so we have to remove english subspace
                    lang_subspace = self.source_lang_subspace  #(B, K, D) . cL^T
                else:
                    lang_subspace = self.target_lang_subspace  #(B, K, D) . cL^T
                lang_subspace_tranposed = lang_subspace.transpose(1, 2) #(B, D, K). c_L

                proj = (lang_subspace_tranposed @ (lang_subspace @ sentence_rep.unsqueeze(2))).squeeze() / sentence_rep_l2norm # language-specific components
                
                hidden_states = hidden_states - proj.unsqueeze(1).expand(hidden_states.shape) * attention_mask_reshaped
            else:
                raise ValueError("Invalid removal type!")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )








class LangAgnosticEncoderwithAdapter(MT5StackwithAdapter):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [MT5BlockWithAdapter(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        lang_ids=None,
        task_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    lang_ids=lang_ids,
                    task_ids=task_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)


        # Remove language identity information here

        assert self.removal_type is not None, "Please specify the removal type"
        random_number = random.random()
        if random_number < self.remove_prob:
            if self.removal_type == "mean_zero":
                if self.is_synthesize:
                    # During inference, we feed english mr as an input - so we have to subtract english mean
                    lang_mean = self.lang_means[torch.zeros_like(lang_ids)]
                else:
                    lang_mean = self.lang_means[lang_ids]
                hidden_states = hidden_states - lang_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            elif self.removal_type == "mean_eng":
                # Shift to English mean, if not English && not pad
                if 0 not in lang_ids:
                    eng_mean = self.lang_means[torch.zeros_like(lang_ids)]
                    lang_mean = self.lang_means[lang_ids]
                    hidden_states = hidden_states - lang_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape) + eng_mean.unsqueeze(1).expand(hidden_states.shape) * attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            elif self.removal_type == "proj_remove":
                attention_mask_reshaped = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
                sentence_rep = torch.sum(hidden_states * attention_mask_reshaped, dim=1) / torch.clamp(attention_mask_reshaped.sum(1), min=1e-9) # We ignore the padded tokens (B, D)
                sentence_rep_l2norm = torch.norm(sentence_rep, dim=-1, p=2, keepdim=True) #(B, 1)
                if self.is_synthesize:
                    # During inference, we feed english mr as an input - so we have to remove english subspace
                    lang_subspace = self.source_lang_subspace  #(B, K, D) . cL^T
                else:
                    lang_subspace = self.target_lang_subspace  #(B, K, D) . cL^T
                lang_subspace_tranposed = lang_subspace.transpose(1, 2) #(B, D, K). c_L

                proj = (lang_subspace_tranposed @ (lang_subspace @ sentence_rep.unsqueeze(2))).squeeze() / sentence_rep_l2norm # language-specific components
                
                hidden_states = hidden_states - proj.unsqueeze(1).expand(hidden_states.shape) * attention_mask_reshaped
            else:
                raise ValueError("Invalid removal type!")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )



class MT5ForConditionalGenerationWithAdapter(MT5ForConditionalGeneration):
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5StackwithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5StackwithAdapter(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.parallelize
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_encoder
    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        in_lang_ids: torch.LongTensor = None,
        out_lang_ids: torch.LongTensor = None,
        task_ids: torch.LongTensor = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5ForConditionalGeneration.from_pretrained("mt5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=in_lang_ids,
                task_ids=task_ids,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            lang_ids=out_lang_ids,
            task_ids=task_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def setup_adapter(self, adapter_config):

        ### Encoder adapter setting
        for i, _ in enumerate(self.encoder.block):
            self.encoder.block[i].config.lang_dict = adapter_config.lang_dict
            self.encoder.block[i].config.task_dict = adapter_config.task_dict

            if adapter_config.adapters["encoder"]["lang"]:
                self.encoder.block[i].Adapter_dict["lang"] = nn.ModuleDict({})
                for lang_id, lang in adapter_config.lang_dict.items():
                    self.encoder.block[i].Adapter_dict["lang"][lang] = Adapter(self.config, adapter_config.adapter_reduction_factor, adapter_config.adapter_layer_norm)

            if adapter_config.adapters["encoder"]["task"]:
                self.encoder.block[i].Adapter_dict["task"] = nn.ModuleDict({})
                for task_id, task in adapter_config.task_dict.items():
                    self.encoder.block[i].Adapter_dict["task"][task] = Adapter(self.config, adapter_config.adapter_reduction_factor, adapter_config.adapter_layer_norm)                

        ### Decoder adapter setting
        for i, _ in enumerate(self.decoder.block):
            self.decoder.block[i].config.lang_dict = adapter_config.lang_dict
            self.decoder.block[i].config.task_dict = adapter_config.task_dict

            if adapter_config.adapters["decoder"]["lang"]:
                self.decoder.block[i].Adapter_dict["lang"] = nn.ModuleDict({})
                for lang_id, lang in adapter_config.lang_dict.items():
                    self.decoder.block[i].Adapter_dict["lang"][lang] = Adapter(self.config, adapter_config.adapter_reduction_factor, adapter_config.adapter_layer_norm)

            if adapter_config.adapters["decoder"]["task"]:
                self.decoder.block[i].Adapter_dict["task"] = nn.ModuleDict({})
                for task_id, task in adapter_config.task_dict.items():
                    self.decoder.block[i].Adapter_dict["task"][task] = Adapter(self.config, adapter_config.adapter_reduction_factor, adapter_config.adapter_layer_norm)

    def freeze_components(self, frozen_list, freeze_lang=False, freeze_task=False):
        # Find the last block of encoder 
        param_list = [name for name, param in self.named_parameters()]
        encoder_final_lm_index = param_list.index("encoder.final_layer_norm.weight")
        encoder_final_block_name = param_list[encoder_final_lm_index-1].split("layer")[0]

        print("Frozen components:", frozen_list)
        for name, param in self.named_parameters():
            if "lang" in name:
                if freeze_lang:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                continue
            if "task" in name:
                if freeze_task:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                continue

            if "all" in frozen_list: ### Freeze all
                param.requires_grad = False
                continue            
            
            if "emb" in frozen_list:
                if "shared" in name or "embed_tokens" in name:
                    param.requires_grad = False


            if "enc_attn" in frozen_list and "encoder" in name and "layer.0" in name: ### Encoder SelfAttention
                param.requires_grad = False
            if "enc_ffn" in frozen_list and "encoder" in name and "layer.1" in name: ### Encoder FeedForward
                param.requires_grad = False
            
            if "enc_lm" in frozen_list and "encoder" in name and "layer_norm" in name: ### Encoder Final layer norm
                param.requires_grad = False
            if "enc_final_lm" in frozen_list and "encoder" in name and "final_layer" in name: ### Encoder Final layer norm
                param.requires_grad = False

            if "enc_final_block" in frozen_list and encoder_final_block_name in name: ### Encoder Final block
                param.requires_grad = False
                
            if "dec_attn" in frozen_list and "decoder" in name and "layer.0" in name: ### Decoder SelfAttention
                param.requires_grad = False
            if "dec_crossattn" in frozen_list and "decoder" in name and "layer.1" in name: ### Encoder CrossAttention
                param.requires_grad = False
            if "dec_ffn" in frozen_list and "decoder" in name and "layer.2" in name: ### Encoder FeedForward
                param.requires_grad = False
            
            if "dec_lm" in frozen_list and "decoder" in name and "layer_norm" in name: ### Encoder Final layer norm
                param.requires_grad = False
            
            if "dec_final" in frozen_list and "decoder" in name and "final_layer" in name: ### Encoder Final layer norm
                param.requires_grad = False

            if "lm_head" in frozen_list and "lm_head" in name:
                param.requires_grad = False
                

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "out_lang_ids"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        if "in_lang_ids" in model_kwargs.keys():
            encoder_kwargs["lang_ids"] = model_kwargs["in_lang_ids"]
        if "task_ids" in model_kwargs.keys():
            encoder_kwargs["task_ids"] = model_kwargs["task_ids"]

        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if "out_lang_ids" in kwargs.keys():
            out_lang_ids = kwargs["out_lang_ids"]
        else:
            out_lang_ids = None

        if "task_ids" in kwargs.keys():
            task_ids = kwargs["task_ids"]
        else:
            task_ids = None

        return {
            "decoder_input_ids": input_ids,
            "out_lang_ids": out_lang_ids,
            "task_ids": task_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @torch.no_grad()
    def generate(
        self,
        inputs = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        **kwargs,
    ):
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

class MT5ForCondGenWithLangAgnosticEncoderWithAdapter(MT5ForConditionalGenerationWithAdapter):
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        
        self.model_dim = config.d_model
    
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LangAgnosticEncoderwithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5StackwithAdapter(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def add_language_identity(self, lang_means, source_lang_subspace, target_lang_subspace, removal_type, remove_prob, is_synthesize):
        # lang_means : (L, D), lang_subspaces: (L, k, D)
        # where L is the number of languages and k is the number of principal components, D is the hidden size
        self.encoder.register_buffer('lang_means', lang_means)

        if source_lang_subspace is not None:
            self.encoder.register_buffer('source_lang_subspace', source_lang_subspace)
            self.encoder.register_buffer('target_lang_subspace', target_lang_subspace)
        self.encoder.removal_type = removal_type
        self.encoder.remove_prob = remove_prob

        self.encoder.is_synthesize = is_synthesize

        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        in_lang_ids: torch.LongTensor = None,
        out_lang_ids: torch.LongTensor = None,
        task_ids: torch.LongTensor = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5ForConditionalGeneration.from_pretrained("mt5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""

        if in_lang_ids is not None and out_lang_ids is not None:
            assert torch.eq(in_lang_ids[0], out_lang_ids[0]), "In-Out lang should be same!"

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=in_lang_ids,
                task_ids=task_ids,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            lang_ids=out_lang_ids,
            task_ids=task_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
"""
For Lora

"""

class MT5ForCondGenWithLangAgnosticEncoderforLoRA(MT5ForConditionalGeneration):
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        
        self.model_dim = config.d_model
    
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LangAgnosticEncoder(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def add_language_identity(self, lang_means, source_lang_subspace, target_lang_subspace, removal_type, remove_prob, is_synthesize):
        # lang_means : (L, D), lang_subspaces: (L, k, D)
        # where L is the number of languages and k is the number of principal components, D is the hidden size
        self.encoder.register_buffer('lang_means', lang_means)

        if source_lang_subspace is not None:
            self.encoder.register_buffer('source_lang_subspace', source_lang_subspace)
            self.encoder.register_buffer('target_lang_subspace', target_lang_subspace)
        self.encoder.removal_type = removal_type
        self.encoder.remove_prob = remove_prob

        self.encoder.is_synthesize = is_synthesize

        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        in_lang_ids: torch.LongTensor = None,
        out_lang_ids: torch.LongTensor = None,
        task_ids: torch.LongTensor = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5ForConditionalGeneration.from_pretrained("mt5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=in_lang_ids,
                task_ids=task_ids,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )