import random

import torch

from train_context import TrainContext
from utils import length_to_mask, log_norm


class BatchContext:
    def __init__(
        self,
        train: TrainContext,
        model,
        texts: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        self.train = train
        self.config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.text_mask = length_to_mask(text_lengths).to(self.config.training.device)
        self.duration_results = None

    def text_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_encoder(texts, text_lengths, self.text_mask)

    def acoustic_duration(
        self,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
        texts: torch.Tensor,
        text_lengths: torch.Tensor,
        apply_attention_mask: bool = False,
        use_random_choice: bool = False,
    ) -> torch.Tensor:
        """
        Computes the duration used for training using a text aligner on
        the combined ground truth audio and text.
        Returns:
          - duration: Duration attention vector
        """
        # Create masks.
        mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
            train.config.training.device
        )

        # --- Text Aligner Forward Pass ---
        with train.accelerator.autocast():
            s2s_pred, s2s_attn = train.model.text_aligner(mels, mask, texts)
            # Remove the last token to make the shape match texts
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

        # Optionally apply extra attention mask.
        if apply_attention_mask:
            with torch.no_grad():
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], self.text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(
                        self.text_mask.shape[0], self.text_mask.shape[1], mask.shape[-1]
                    )
                    .float()
                )
                attn_mask = attn_mask < 1
            s2s_attn.masked_fill_(attn_mask, 0.0)

        # --- Monotonic Attention Path ---
        with torch.no_grad():
            mask_ST = mask_from_lens(
                s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # --- Text Encoder Forward Pass ---
        with train.accelerator.autocast():
            if use_random_choice and bool(random.getrandbits(1)):
                duration = s2s_attn
            else:
                duration = s2s_attn_mono

        self.duration_results = (s2s_attn, s2s_attn_mono)
        return duration

    def acoustic_pitch(self, mels: torch.Tensor):
        with torch.no_grad():
            pitch, _, _ = self.model.pitch_extractor(mels.unsqueeze(1))
        return pitch

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def acoustic_style_embedding(self, mels: torch.Tensor):
        return self.model.style_encoder(mels.unsqueeze(1))

    def decoding(
        self, text_encoding, duration, pitch, energy, style, audio_gt, split=1
    ):
        if split == 1 or text_encoding.shape[0] != 1:
            audio_out, mag, phase = self.model.decoder(
                text_encoding @ duration, pitch, energy, style
            )
            yield (audio_out, mag, phase, audio_gt)
        else:
            text_hop = text_encoding.shape[-1] // split
            text_start = 0
            text_end = text_hop + text_encoding.shape[-1] % split
            mel_start = 0
            mel_end = 0
            for i in range(split):
                mel_hop = int(duration[:, text_start:text_end, :].sum().item())
                mel_start = mel_end
                mel_end = mel_start + mel_hop

                text_slice = text_encoding[:, :, text_start:text_end]
                duration_slice = duration[:, text_start:text_end, mel_start:mel_end]
                pitch_slice = pitch[:, mel_start * 2 : mel_end * 2]
                energy_slice = energy[:, mel_start * 2 : mel_end * 2]
                audio_gt_slice = audio_gt[
                    :, mel_start * 300 * 2 : mel_end * 300 * 2
                ].detach()
                audio_out, mag, phase = train.model.decoder(
                    text_slice @ dur_slice, pitch_slice, energy_slice, style
                )
                yield (audio_out, mag, phase, audio_gt_slice)
                text_start += text_hop
                text_end += text_hop
