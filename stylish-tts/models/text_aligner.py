"""
Text Aligner

- Taken from StyleTTS 2 repo here: https://github.com/yl4579/StyleTTS2
- Training code is here: https://github.com/yl4579/AuxiliaryASR
- Paper sources the model here: https://zenodo.org/records/4541452
"""

import math
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F
from .text_aligner_layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock
from einops import rearrange


def tdnn_blstm_ctc_model(
    input_dim: int, num_symbols: int, hidden_dim=640, drop_out=0.1, tdnn_blstm_spec=[]
):
    r"""Builds TDNN-BLSTM-based CTC model."""
    encoder = TdnnBlstm(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        drop_out=drop_out,
        tdnn_blstm_spec=tdnn_blstm_spec,
    )
    encoder_output_layer = nn.Linear(hidden_dim, num_symbols + 1)

    return CTCModel(encoder, encoder_output_layer)


def tdnn_blstm_ctc_model_base(n_mels, num_symbols):
    return tdnn_blstm_ctc_model(
        input_dim=n_mels,
        num_symbols=num_symbols,
        hidden_dim=640,
        drop_out=0.1,
        tdnn_blstm_spec=[
            ("tdnn", 5, 2, 1),
            ("tdnn", 3, 1, 1),
            ("tdnn", 3, 1, 1),
            ("ffn", 5),
        ],
    )


class CTCModel(torch.nn.Module):
    r"""
    This implements a CTC model with an encoder and a projection layer
    """

    def __init__(self, encoder, encoder_output_layer) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_output_layer = encoder_output_layer

    def ctc_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The output tensor from the transformer encoder.
            Its shape is (B, T', D')

        Returns:
          Return a tensor that can be used for CTC decoding.
          Its shape is (B, T, V), where V is the number of classes
        """
        x = self.encoder_output_layer(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (T, N, C)
        return x

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
        """
        device = self.encoder_output_layer.weight.device
        if sources.device != device:
            sources = sources.to(device)
            source_lengths = source_lengths.to(device)

        source_encodings, source_lengths = self.encoder(
            input=sources,
            lengths=source_lengths,
        )

        ctc_log_prob = self.ctc_output(source_encodings)

        return ctc_log_prob


class TdnnBlstm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim=640,
        drop_out=0.1,
        tdnn_blstm_spec=[],
    ) -> None:
        """
        Args:

          tdnn_blstm_spec:
            It is a list of network specifications. It can be either:
            - ('tdnn', kernel_size, stride, dilation)
            - ('blstm')
        """
        super().__init__()

        self.tdnn_blstm_spec = tdnn_blstm_spec

        layers = nn.ModuleList([])
        layers_info = []
        for i_layer, spec in enumerate(tdnn_blstm_spec):
            if spec[0] == "tdnn":
                ll = []
                dilation = spec[3] if len(spec) >= 4 else 1
                padding = int((spec[1] - 1) / 2) * dilation
                ll.append(
                    nn.Conv1d(
                        in_channels=input_dim if len(layers) == 0 else hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=spec[1],  # 3
                        dilation=dilation,
                        stride=spec[2],  # 1
                        padding=padding,  # 1
                    )
                )
                ll.append(nn.ReLU(inplace=True))
                ll.append(nn.BatchNorm1d(num_features=hidden_dim, affine=False))
                if drop_out > 0:
                    ll.append(nn.Dropout(drop_out))

                # The last dimension indicates the stride size
                # If stride > 1, then we need to recompute the lengths of input after this layer
                layers.append(nn.Sequential(*ll))
                layers_info.append(("tdnn", spec))

            elif spec[0] == "blstm":
                layers.append(
                    Blstm_with_skip(
                        input_dim=input_dim if len(layers) == 0 else hidden_dim,
                        hidden_dim=hidden_dim,
                        out_dim=hidden_dim,
                        skip=(
                            False
                            if len(layers) == 0 and input_dim != hidden_dim
                            else True
                        ),
                        drop_out=drop_out,
                    )
                )
                layers_info.append(("blstm", None))

            elif spec[0] == "ffn":
                layers.append(
                    Ffn(
                        input_dim=input_dim if len(layers) == 0 else hidden_dim,
                        hidden_dim=hidden_dim,
                        out_dim=hidden_dim,
                        skip=True,
                        drop_out=drop_out,
                        nlayers=spec[1],
                    )
                )
                layers_info.append(("ffn", spec))

        self.layers = layers
        self.layers_info = layers_info

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            Its shape is [N, T, C]

        Returns:
          The output tensor has shape [N, T, C]
        """
        x = input
        for layer, (layer_type, spec) in zip(self.layers, self.layers_info):
            if layer_type == "tdnn":
                mask = (
                    torch.arange(lengths.max(), device=x.device)[None, :]
                    < lengths[:, None]
                ).float()
                x = x * mask.unsqueeze(2)  # masking/padding
                x = x.permute(0, 2, 1)  # (N, T, C) ->(N, C, T)
                x = layer(x)
                x = x.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

                stride = spec[2]
                if True:  # stride > 1:
                    kernel_size = spec[1]
                    dilation = spec[3] if len(spec) >= 4 else 1
                    padding = int((spec[1] - 1) / 2) * dilation
                    lengths = lengths + 2 * padding - dilation * (kernel_size - 1) - 1
                    lengths = lengths / stride + 1
                    lengths = torch.floor(lengths)
            elif layer_type == "blstm":
                x = layer(x, lengths)
            else:
                x = layer(x)
        return x, lengths


class Ffn(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, out_dim, nlayers=1, drop_out=0.1, skip=False
    ) -> None:
        super().__init__()

        layers = []
        for ilayer in range(nlayers):
            _in = hidden_dim if ilayer > 0 else input_dim
            _out = hidden_dim if ilayer < nlayers - 1 else out_dim
            layers.extend(
                [
                    nn.Linear(_in, _out),
                    nn.ReLU(),
                    nn.Dropout(p=drop_out),
                ]
            )
        self.ffn = torch.nn.Sequential(
            *layers,
        )

        self.skip = skip

    def forward(self, x) -> torch.Tensor:
        x_out = self.ffn(x)

        if self.skip:
            x_out = x_out + x

        return x_out


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
