import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, maximum_path, print_gpu_vram


class BatchContext:
    def __init__(
        self,
        *,
        train: train_context.TrainContext,
        model,
        text_length: Optional[torch.Tensor],
    ):
        self.train: train_context.TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.text_mask = None
        if text_length is not None:
            self.text_mask = length_to_mask(text_length).to(self.config.training.device)
        self.duration_results = None
        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

    def text_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_encoder(texts, text_lengths)

    def text_duration_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_duration_encoder(texts, text_lengths)

    def bert_encoding(self, texts: torch.Tensor):
        mask = (~self.text_mask).int()
        bert_encoding = self.model.bert(texts, attention_mask=mask)
        text_encoding = self.model.bert_encoder(bert_encoding)
        return text_encoding.transpose(-1, -2)

    def acoustic_duration(
        self,
        batch,
    ) -> torch.Tensor:
        """
        Computes the duration used for training using a text aligner on
        the combined ground truth audio and text.
        Returns:
          - duration: Duration attention vector
        """
        duration = batch.alignment
        self.attention = duration[0]
        self.duration_results = (duration, duration)
        return duration

    def get_attention(self):
        return self.attention

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def calculate_pitch(self, batch, prediction=None):
        if prediction is None:
            prediction = batch.pitch
        return prediction

    def acoustic_style_embedding(self, mels: torch.Tensor):
        return self.model.acoustic_style_encoder(mels.unsqueeze(1))

    def acoustic_prosody_embedding(self, mels: torch.Tensor):
        return self.model.acoustic_prosody_encoder(mels.unsqueeze(1))

    def textual_style_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_style_encoder(sentence_embedding)

    def textual_prosody_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_prosody_encoder(sentence_embedding)

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        probing=False,
    ):
        mel, f0_curve = self.model.decoder(
            text_encoding @ duration, pitch, energy, style @ duration, probing=probing
        )
        print_gpu_vram("mel_decoder")
        result = self.model.generator(
            mel=mel, style=style @ duration, pitch=f0_curve, energy=energy
        )
        print_gpu_vram("generator")
        return result

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        text_encoding, _, _ = self.text_encoding(batch.text, batch.text_length)
        print_gpu_vram("text encoder")
        duration = self.acoustic_duration(
            batch,
        )
        energy = self.acoustic_energy(batch.mel)
        style_embedding = self.textual_style_embedding(text_encoding)
        print_gpu_vram("style")
        pitch = self.calculate_pitch(batch).detach()
        prediction = self.decoding_single(
            text_encoding,
            duration,
            pitch,
            energy,
            style_embedding,
        )
        return prediction

    def textual_prediction_single(self, batch):
        text_encoding, _, _ = self.text_encoding(batch.text, batch.text_length)
        duration_encoding, _, _ = self.text_duration_encoding(
            batch.text, batch.text_length
        )
        duration = self.acoustic_duration(
            batch,
        )
        style_embedding = self.textual_style_embedding(text_encoding)
        prosody_embedding = self.textual_prosody_embedding(duration_encoding)
        self.duration_prediction, prosody = self.model.duration_predictor(
            duration_encoding,
            prosody_embedding,
            batch.text_length,
            duration,
            self.text_mask,
        )
        prosody_embedding = prosody_embedding @ duration
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(prosody, prosody_embedding)
        )
        pitch = self.calculate_pitch(batch, self.pitch_prediction)
        prediction = self.decoding_single(
            text_encoding,
            duration,
            pitch,
            self.energy_prediction,
            style_embedding,
        )
        return prediction
