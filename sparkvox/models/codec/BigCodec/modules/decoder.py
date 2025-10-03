# Taken from https://github.com/Aria-K-Alethia/BigCodec/blob/main/vq/codec_decoder.py

import numpy as np
import torch
import torch.nn as nn

from sparkvox.models.codec.BigCodec.modules import activations
from sparkvox.models.codec.BigCodec.modules.alias_free_torch import *
from sparkvox.models.codec.BigCodec.modules.module import (
    WNConv1d,
    DecoderBlock,
    ResLSTM,
)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        upsample_initial_channel=1536,
        ngf=48,
        use_rnn=True,
        rnn_bidirectional=False,
        rnn_num_layers=2,
        up_ratios=(5, 5, 2, 2, 2),
        dilations=(1, 3, 9),
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]

        if use_rnn:
            layers += [
                ResLSTM(
                    channels, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional
                )
            ]

        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations)]

        layers += [
            Activation1d(
                activation=activations.SnakeBeta(output_dim, alpha_logscale=True)
            ),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.linear_pre(x.transpose(1,2))
        x = self.downsample(x).transpose(1,2)
        x = self.vocos_backbone(x, condition=c)
        x = self.linear(x).transpose(1,2)
        if self.use_tanh_at_final:
            x = torch.tanh(x)

    def inference_vq(self, vq):
        x = vq[None, :, :]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x):
        x = self.model(x)
        return x, None

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)