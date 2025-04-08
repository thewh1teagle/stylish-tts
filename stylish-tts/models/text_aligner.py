"""
Text Aligner

- Taken from StyleTTS 2 repo here: https://github.com/yl4579/StyleTTS2
- Training code is here: https://github.com/yl4579/AuxiliaryASR
- Paper sources the model here: https://zenodo.org/records/4541452
"""

import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F
from .text_aligner_layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock
from einops import rearrange


# from loss import *
import functools
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from einops import rearrange
from scipy import ndimage
from torch.special import gammaln
import torch.nn as nn

# from utils import *


class AlignmentEncoder(torch.nn.Module):
    """
    Module for alignment text and mel spectrogram.

    Args:
        n_mel_channels: Dimension of mel spectrogram.
        n_text_channels: Dimension of text embeddings.
        n_att_channels: Dimension of model
        temperature: Temperature to scale distance by.
            Suggested to be 0.0005 when using dist_type "l2" and 15.0 when using "cosine".
        condition_types: List of types for nemo.collections.tts.modules.submodules.ConditionalInput.
        dist_type: Distance type to use for similarity measurement. Supports "l2" and "cosine" distance.
    """

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512,
        n_att_channels=128,
        temperature=0.0005,
        condition_types=[],
        dist_type="l2",
    ):
        super().__init__()
        self.temperature = temperature
        # self.cond_input = ConditionalInput(n_text_channels, n_text_channels, condition_types)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )
        if dist_type == "l2":
            self.dist_fn = self.get_euclidean_dist
        elif dist_type == "cosine":
            self.dist_fn = self.get_cosine_dist
        else:
            raise ValueError(f"Unknown distance type '{dist_type}'")

    @staticmethod
    def _apply_mask(inputs, mask, mask_value):
        if mask is None:
            return

        mask = rearrange(mask, "B T2 1 -> B 1 1 T2")
        inputs.data.masked_fill_(mask, mask_value)

    def get_dist(self, keys, queries, mask=None):
        """Calculation of distance matrix.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries and also can be used
                for ignoring unnecessary elements from keys in the resulting distance matrix (True = mask element, False = leave unchanged).
        Output:
            dist (torch.tensor): B x T1 x T2 tensor.
        """
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        dist = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)

        self._apply_mask(dist, mask, float("inf"))

        return dist.squeeze(1)

    @staticmethod
    def get_euclidean_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        # B x C x T1 x T2
        distance = (queries_enc - keys_enc) ** 2
        # B x 1 x T1 x T2
        l2_dist = distance.sum(axis=1, keepdim=True)
        return l2_dist

    @staticmethod
    def get_cosine_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        cosine_dist = -torch.nn.functional.cosine_similarity(
            queries_enc, keys_enc, dim=1
        )
        cosine_dist = rearrange(cosine_dist, "B T1 T2 -> B 1 T1 T2")
        return cosine_dist

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.

        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.
        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and return mean distance.

        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.
        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        AlignmentEncoder._apply_mask(dist, mask, 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(
                            torch.arange(t2_size), repeats=durations[dist_idx]
                        ),
                    ]
                )
            )

        return torch.tensor(
            mean_dist_by_durations, dtype=dist.dtype, device=dist.device
        )

    @staticmethod
    def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
        """Calculates the mean distance between text and audio embeddings given a range of text tokens.

        Args:
            l2_dists (torch.tensor): L2 distance matrix from Aligner inference. T1 x T2 tensor.
            durs (torch.tensor): List of durations corresponding to each text token. T2 tensor. Should sum to T1.
            start_token (int): Index of the starting token for the word of interest.
            num_tokens (int): Length (in tokens) of the word of interest.
        Output:
            mean_dist_for_word (float): Mean embedding distance between the word indicated and its predicted audio frames.
        """
        # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
        start_frame = torch.sum(durs[:start_token]).data

        total_frames = 0
        dist_sum = 0

        # Loop through each text token
        for token_ind in range(start_token, start_token + num_tokens):
            # Loop through each frame for the given text token
            for frame_ind in range(start_frame, start_frame + durs[token_ind]):
                # Recall that the L2 distance matrix is shape [spec_len, text_len]
                dist_sum += l2_dists[frame_ind, token_ind]

            # Update total frames so far & the starting frame for the next token
            total_frames += durs[token_ind]
            start_frame += durs[token_ind]

        return dist_sum / total_frames

    def forward(self, queries, keys, mask=None, attn_prior=None, conditioning=None):
        """Forward pass of the aligner encoder.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries (True = mask element, False = leave unchanged).
            attn_prior (torch.tensor): prior for attention matrix.
            conditioning (torch.tensor): B x 1 x C2 conditioning embedding
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        # keys = self.cond_input(keys.transpose(1, 2), conditioning).transpose(1, 2)
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        distance = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)
        attn = -self.temperature * distance

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        self._apply_mask(attn, mask, -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


def get_mask_from_lengths(
    lengths: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths: Optional[torch.tensor] (torch.tensor): 1D tensor with lengths
        x: Optional[torch.tensor] = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


class TextAligner(torch.nn.Module):
    """Speech-to-text alignment model (https://arxiv.org/pdf/2108.10447.pdf) that is used to learn alignments between mel spectrogram and text."""

    def __init__(self):

        # num_tokens = len(self.tokenizer.tokens)
        # self.tokenizer_pad = self.tokenizer.pad
        # self.tokenizer_unk = self.tokenizer.oov

        super().__init__()

        self.embed = nn.Embedding(178, 512)
        self.alignment_encoder = AlignmentEncoder()

        # self.bin_loss = BinLoss()
        # self.add_bin_loss = False
        # self.bin_loss_scale = 0.0
        # self.bin_loss_start_ratio = cfg.bin_loss_start_ratio
        # self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

    def forward(self, *, spec, text, text_len, attn_prior=None):
        # with torch.amp.autocast(self.device.type, enabled=False):
        attn_soft, attn_logprob = self.alignment_encoder(
            queries=spec,
            keys=self.embed(text).transpose(1, 2),
            mask=get_mask_from_lengths(text_len).unsqueeze(-1) == 0,
            attn_prior=attn_prior,
        )
        attn_logprob = rearrange(attn_logprob, "b 1 t p -> b t p")
        return attn_soft, attn_logprob


# class TextAligner(nn.Module):
#     def __init__(self, *, n_mels, n_token):
#         super().__init__()
#         self.lstm1 = nn.LSTM(
#             n_mels,
#             128,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.0,
#         )
#         self.lstm2 = nn.LSTM(
#             256,
#             128,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.0,
#         )
#         self.end_encode = nn.Linear(256, n_token + 1, bias=False)
#
#         self.lrelu = nn.LeakyReLU()
#
#         self.decode = nn.Sequential(
#             nn.Linear(n_token, 256, bias=False),
#             nn.LeakyReLU(),
#             nn.Linear(256, 256, bias=False),
#             nn.LeakyReLU(),
#             nn.Linear(256, n_mels, bias=False),
#         )
#
#     def remove_blanks(self, x):
#         """
#         Remove trailing blank column from tensor
#
#         Args:
#             x (b t k): k is tokens + blank
#         """
#         return x[:, :, :-1]
#
#     def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             mel (b f t): Mel spectrogram
#         Returns:
#             prediction (b t k): Softmax point probability of tokens for each mel time frame
#             reconstruction (b f t): Reconstructed mel spectrogram
#         """
#         x = rearrange(mel, "b f t -> b t f")
#         x, _ = self.lstm1(x)
#         x = self.lrelu(x)
#         x, _ = self.lstm2(x)
#         x = self.lrelu(x)
#         x = self.end_encode(x)
#         # x is now (b t k) where k is the # of tokens + blank
#         prediction = x
#
#         x = self.remove_blanks(x)
#         x = self.decode(x)
#         # x is now (b t f) again, a reconstructed mel spectrogram
#         reconstruction = rearrange(x, "b t f -> b f t")
#         return prediction, reconstruction


class TextAlignerOld(nn.Module):
    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        n_token=35,
        n_layers=6,
        token_embedding_dim=256,
    ):
        super().__init__()
        self.n_token = n_token
        self.to_mfcc = MFCC(n_mfcc=input_dim / 2, n_mels=input_dim)
        self.init_cnn = ConvNorm(
            input_dim // 2, hidden_dim, kernel_size=7, padding=3, stride=2
        )
        self.cnns = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(hidden_dim),
                    nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                )
                for n in range(n_layers)
            ]
        )
        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            LinearNorm(hidden_dim, n_token),
        )
        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token,
        )

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.projection(x)
        x = x.transpose(1, 2)
        ctc_logit = self.ctc_linear(x)
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.asr_s2s(x, src_key_padding_mask, text_input)
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return None  # ctc_logit

    def get_feature(self, x):
        x = self.to_mfcc(x.squeeze(1))
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1)).to(lengths.device)
        return mask

    def get_future_mask(self, out_length, unmask_future_steps=0):
        """
        Args:
            out_length (int): returned mask shape is (out_length, out_length).
            unmask_futre_steps (int): unmasking future step size.
        Return:
            mask (torch.BoolTensor): mask future timesteps mask[i, j] = True if i > j + unmask_future_steps else False
        """
        index_tensor = torch.arange(out_length).unsqueeze(0).expand(out_length, -1)
        mask = torch.gt(index_tensor, index_tensor.T + unmask_future_steps)
        return mask


class ASRS2S(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        hidden_dim=512,
        n_location_filters=32,
        location_kernel_size=63,
        n_token=40,
    ):
        super(ASRS2S, self).__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(
            self.decoder_rnn_dim + embedding_dim, self.decoder_rnn_dim
        )
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim), nn.Tanh()
        )
        self.sos = 1
        self.eos = 2

    def initialize_decoder_states(self, memory, mask):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        """
        B, L, H = memory.shape
        self.decoder_hidden = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.decoder_cell = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.attention_weights = torch.zeros((B, L)).type_as(memory)
        self.attention_weights_cum = torch.zeros((B, L)).type_as(memory)
        self.attention_context = torch.zeros((B, H)).type_as(memory)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        self.unk_index = 3
        self.random_mask = 0.1

    def forward(self, memory, memory_mask, text_input):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        moemory_mask.shape = (B, L, )
        texts_input.shape = (B, T)
        """
        self.initialize_decoder_states(memory, memory_mask)
        # text random mask
        random_mask = (torch.rand(text_input.shape) < self.random_mask).to(
            text_input.device
        )
        _text_input = text_input.clone()
        _text_input.masked_fill_(random_mask, self.unk_index)
        decoder_inputs = self.embedding(_text_input).transpose(
            0, 1
        )  # -> [T, B, channel]
        start_embedding = self.embedding(
            torch.LongTensor([self.sos] * decoder_inputs.size(1)).to(
                decoder_inputs.device
            )
        )
        decoder_inputs = torch.cat(
            (start_embedding.unsqueeze(0), decoder_inputs), dim=0
        )

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):
            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]

        hidden_outputs, logit_outputs, alignments = self.parse_decoder_outputs(
            hidden_outputs, logit_outputs, alignments
        )

        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, (self.decoder_hidden, self.decoder_cell)
        )

        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        )

        self.attention_weights_cum += self.attention_weights

        hidden_and_context = torch.cat(
            (self.decoder_hidden, self.attention_context), -1
        )
        hidden = self.project_to_hidden(hidden_and_context)

        # dropout to increasing g
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self, hidden, logit, alignments):
        # -> [B, T_out + 1, max_time]
        alignments = torch.stack(alignments).transpose(0, 1)
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1,  n_symbols]
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()

        return hidden, logit, alignments
