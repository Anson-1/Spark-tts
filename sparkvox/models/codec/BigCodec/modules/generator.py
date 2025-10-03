import torch
import torch.nn as nn

from typing import Dict, Any
from hydra.utils import instantiate
from omegaconf import DictConfig

from sparkvox.models.codec.BigCodec.modules.encoder import Encoder
from sparkvox.models.codec.BigCodec.modules.decoder import Decoder
from sparkvox.models.codec.base.quantize.factorized_vector_quantize import FactorizedVectorQuantize

class Generator(nn.Module):
    """Generator module."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        **kwargs
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        wav = batch["wav"]
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(1)
        z = self.encoder(wav)
        vq_outputs = self.quantizer(z)
        wav_recon = self.decoder(vq_outputs['z_q'])
        
        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "audios": wav,
            "recons": wav_recon
        }

    def remove_weight_norm(self):
        
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
