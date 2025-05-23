# coding:utf-8


import math

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import ReformerLM, Autopadder

from stylish_lib.config_loader import ModelConfig


from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminators.multi_period import MultiPeriodDiscriminator
from .discriminators.multi_resolution import MultiResolutionDiscriminator
from .discriminators.multi_subband import MultiScaleSubbandCQTDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from .text_encoder import TextEncoder
from .fine_style_encoder import FineStyleEncoder
from .decoder.mel_decoder import MelDecoder
from .decoder.freev import FreevGenerator
from .decoder.ringformer import RingformerGenerator

from munch import Munch
import safetensors
from huggingface_hub import hf_hub_download

import logging

logger = logging.getLogger(__name__)


class TextualStyleEncoder(nn.Linear):
    """Linear layer with merciful load_state_dict."""

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict:
            try:
                return super().load_state_dict(state_dict, strict, assign)
            except:
                logger.warning(
                    "The state dict from the checkpoint of TextualStyleEncoder is not compatible. Ignore this message if the sbert is intentionally changed."
                )


def build_model(model_config: ModelConfig, sbert_output_dim):
    text_aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.n_token
    )
    assert model_config.decoder.type in [
        "ringformer",
        "freev",
    ], "Decoder type unknown"

    if model_config.decoder.type == "ringformer":
        decoder = MelDecoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.inter_dim,
        )
        generator = RingformerGenerator(
            style_dim=model_config.style_dim,
            resblock_kernel_sizes=model_config.decoder.resblock_kernel_sizes,
            upsample_rates=model_config.decoder.upsample_rates,
            upsample_initial_channel=model_config.decoder.upsample_initial_channel,
            resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.decoder.gen_istft_n_fft,
            gen_istft_hop_size=model_config.decoder.gen_istft_hop_size,
            sample_rate=model_config.sample_rate,
        )
    else:
        decoder = MelDecoder()
        generator = FreevGenerator()

    text_encoder = TextEncoder(
        n_vocab=model_config.text_encoder.n_token,
        inter_dim=model_config.inter_dim,
        hidden_dim=192,
        filter_channels=768,
        heads=2,
        layers=6,
        kernel_size=3,
        dropout=0.1,
    )
    text_duration_encoder = TextEncoder(
        n_vocab=model_config.text_encoder.n_token,
        inter_dim=model_config.inter_dim,
        hidden_dim=192,
        filter_channels=768,
        heads=2,
        layers=6,
        kernel_size=3,
        dropout=0.1,
    )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        nlayers=model_config.duration_predictor.n_layer,
        max_dur=model_config.duration_predictor.max_dur,
        dropout=model_config.duration_predictor.dropout,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        dropout=model_config.pitch_energy_predictor.dropout,
    )
    nets = Munch(
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        decoder=decoder,
        generator=generator,
        text_encoder=text_encoder,
        text_duration_encoder=text_duration_encoder,
        textual_prosody_encoder=FineStyleEncoder(
            model_config.inter_dim, model_config.style_dim, 4
        ),
        textual_style_encoder=FineStyleEncoder(
            model_config.inter_dim, model_config.style_dim, 4
        ),
        text_aligner=text_aligner,
        mpd=MultiPeriodDiscriminator(),
        msbd=MultiScaleSubbandCQTDiscriminator(sample_rate=model_config.sample_rate),
        mrd=MultiResolutionDiscriminator(),
    )

    return nets
