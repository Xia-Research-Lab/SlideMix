""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re
from typing import Optional

import torch
import torch.nn as nn
from torch import TensorType

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: Optional[PretrainedConfig] = None,
            pooler_type: Optional[str] = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        
        # Load config and tokenizer
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
            
            # Handle different model architectures
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                # For encoder-decoder models, only use the encoder
                self.transformer = (AutoModel.from_pretrained(model_name_or_path) if pretrained 
                                 else AutoModel.from_config(self.config)).encoder
            else:
                # For decoder-only or encoder-only models
                uses_transformer_pooler = (pooler_type == "cls_pooler" and not getattr(self.config, "is_decoder", False))
                kwargs = {"add_pooling_layer": True} if uses_transformer_pooler else {}
                self.transformer = (AutoModel.from_pretrained(model_name_or_path, **kwargs) if pretrained 
                                 else AutoModel.from_config(self.config, **kwargs))
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)

        # Set pooler type based on model architecture
        if pooler_type is None:
            if getattr(self.config, "is_decoder", False):
                pooler_type = "mean_pooler"  # Default to mean pooling for decoder-only models
            else:
                pooler_type = arch_dict.get(self.config.model_type, {}).get("pooler", "mean_pooler")

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        # Get model dimension and create projection
        if self.config.model_type in arch_dict:
            d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        else:
            d_model = getattr(self.config, "hidden_size", getattr(self.config, "d_model", None))
        
        if d_model is None:
            raise ValueError(f"Could not determine model dimension for {model_name_or_path}")

        if (d_model == output_dim) and (proj_type is None):
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(self, x: TensorType):
        attn_mask = (x != self.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        
        # Handle different model architectures
        if isinstance(out, BaseModelOutputWithPooling) and out.pooler_output is not None:
            # Use pooler output if available (e.g., BERT)
            pooled = out.pooler_output
        else:
            # Otherwise use custom pooling
            pooled = self.pooler(out, attn_mask)
            
        projected = self.proj(pooled)

        if self.output_tokens:
            # Get tokens excluding pooling token if present
            if hasattr(self.pooler, 'cls_token_position'):
                seq_len = out.last_hidden_state.shape[1]
                tokens = out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :]
            else:
                tokens = out.last_hidden_state
            return projected, tokens
            
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
