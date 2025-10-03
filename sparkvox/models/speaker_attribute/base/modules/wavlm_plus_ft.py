# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)

# the code is mainly based on
# https://github.com/usc-sail/peft-ser/blob/74cd9e2c8eafac577a49b792f26694bcde5664ad/package/peft_ser/model/wavlm_plus.py#L92
# whic is referenced from:
# SUPERB: https://github.com/s3prl/s3prl
# https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py


import torch
import loralib as lora
import transformers.models.wavlm.modeling_wavlm as wavlm

from torch import nn
from torch.nn import functional as F
from transformers import WavLMModel

from sparkvox.models.speaker_attribute.base.modules.adapter import Adapter


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.config = config

        if (
            self.config.finetune_method == "embedding_prompt"
            or self.config.finetune_method == "combined"
        ):
            self.embed_prompt = nn.Parameter(
                torch.randn([1, self.config.embedding_prompt_dim, 768])
            )
            nn.init.xavier_uniform_(self.embed_prompt)
        if (
            self.config.finetune_method == "lora"
            or self.config.finetune_method == "combined"
        ):
            self.feed_forward.intermediate_dense = lora.Linear(
                config.hidden_size, config.intermediate_size, r=config.lora_rank
            )
            self.feed_forward.output_dense = lora.Linear(
                config.intermediate_size, config.hidden_size, r=config.lora_rank
            )

        if (
            self.config.finetune_method == "adapter"
            or self.config.finetune_method == "adapter_l"
            or self.config.finetune_method == "combined"
        ):
            self.adapter = Adapter(
                config,
                dropout=0.1,
                bottleneck=config.adapter_hidden_dim,
                adapter_scalar=0.1,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        index=0,
    ):
        if (
            self.config.finetune_method == "embedding_prompt"
            or self.config.finetune_method == "combined"
        ):
            hidden_states = torch.cat(
                (self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states),
                dim=1,
            )

        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        if self.config.finetune_method == "adapter":
            hidden_states = hidden_states + adapt_h
        if (
            self.config.finetune_method == "adapter_l"
            or self.config.finetune_method == "combined"
        ):
            hidden_states = hidden_states + self.adapter(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        if (
            self.config.finetune_method == "embedding_prompt"
            or self.config.finetune_method == "combined"
        ):
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim :, :]
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WavLMWrapper(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "microsoft/wavlm-base-plus",
        hidden_dim: int = 768,
        output_class_num: int = 4,
        use_conv_output: bool = True,
        **kwargs,
    ):
        super(WavLMWrapper, self).__init__()
        # Load the model first with weights
        self.backbone_model = WavLMModel.from_pretrained(
            pretrained_model, output_hidden_states=True
        )
        # Read the model config
        self.model_config = self.backbone_model.config

        self.use_conv_output = use_conv_output
        if use_conv_output:
            num_layers = (
                self.model_config.num_hidden_layers + 1
            )  # transformer layers + input embeddings
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
      
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, x: torch.Tensor, length: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): [B, T]
            length (torch.Tensor): [B], x length
        """
        device = x.device
        x = self.backbone_model.feature_extractor(x)
        x = x.transpose(
            1, 2
        )  # New version of huggingface. The final x is with size [batch_size, feature_length, feature_dim]
        x, _ = self.backbone_model.feature_projection(
            x
        )  # New version of huggingface

        # get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.to(device)

        # transformer encoding features
        x = self.backbone_model.encoder(x, output_hidden_states=True).hidden_states

        # stacked feature
        if self.use_conv_output:
            stacked_feature = torch.stack(x, dim=0)
        else:
            stacked_feature = torch.stack(x, dim=0)[1:]

        # Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers + 1, -1
            )
        else:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers, -1
            )
        norm_weights = F.softmax(self.layer_weights, dim=-1)

        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        # Pooling
        if length is not None:
            masks = torch.arange(features.size(1)).expand(length.size(0), -1).to(
                device
            ) < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)

        # Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted

    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(
            self.backbone_model.config.conv_kernel,
            self.backbone_model.config.conv_stride,
        ):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length


# test
if __name__ == "__main__":
    model = WavLMWrapper(
        hidden_dim=768,
        output_class_num=4,
        use_conv_output=True,
    )
    data = torch.zeros([1, 16000])
    length = torch.tensor([16000])
    output = model(data, length)
    print(output.shape)
