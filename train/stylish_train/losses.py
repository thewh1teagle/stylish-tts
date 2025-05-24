import math
from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
import numpy as np
import k2


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        *,
        fft_size,
        shift_size,
        win_length,
        window,
        n_mels,
        sample_rate,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )

        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std

        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std

        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        return sc_loss


class Resolution:
    def __init__(self, *, fft, hop, window, mels):
        self.fft = fft
        self.hop = hop
        self.window = window
        self.mels = mels


resolutions = [
    # Resolution(fft=256, hop=31, window=67, mels=40),
    # Resolution(fft=256, hop=67, window=127, mels=40),
    # Resolution(fft=512, hop=127, window=257, mels=80),
    # Resolution(fft=1024, hop=257, window=509, mels=120),
    # Resolution(fft=2048, hop=509, window=1021, mels=120),
    # Resolution(fft=4096, hop=1021, window=2053, mels=120),
    Resolution(fft=1024, hop=120, window=600, mels=128),
    Resolution(fft=2048, hop=240, window=1200, mels=128),
    Resolution(fft=512, hop=50, window=240, mels=128),
]


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        *,
        resolution_list=resolutions,
        window=torch.hann_window,
        sample_rate,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = torch.nn.ModuleList()
        for item in resolution_list:
            self.stft_losses += [
                STFTLoss(
                    fft_size=item.fft,
                    shift_size=item.hop,
                    win_length=item.window,
                    window=window,
                    sample_rate=sample_rate,
                    n_mels=item.mels,
                )
            ]

    def forward(self, x, y, log):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_loss += f(x, y)
        sc_loss /= len(self.stft_losses)

        log.add_loss("mel", sc_loss)

        return sc_loss


class MagPhaseLoss(torch.nn.Module):
    """Magnitude/Phase Loss for Ringformer"""

    def __init__(self, *, n_fft, hop_length):
        super(MagPhaseLoss, self).__init__()
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, mag, phase, gt):
        result = 0.0
        if mag is not None and phase is not None:
            y_stft = torch.stft(
                gt,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                return_complex=True,
                window=self.window,
            )
            target_mag = torch.abs(y_stft)
            target_phase = torch.angle(y_stft)
            mag_loss = torch.nn.functional.l1_loss(mag, target_mag)
            phase_loss = torch.nn.functional.l1_loss(phase, target_phase)
            result = mag_loss + phase_loss
        return result


def amplitude_loss(log_amplitude_r, log_amplitude_g):
    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_loss(phase_r, phase_g, n_fft, frames):
    GD_matrix = (
        torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        - torch.eye(n_fft // 2 + 1)
    )
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0, 2, 1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0, 2, 1), GD_matrix)

    PTD_matrix = (
        torch.triu(torch.ones(frames, frames), diagonal=1)
        - torch.triu(torch.ones(frames, frames), diagonal=2)
        - torch.eye(frames)
    )
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r - GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r - PTD_g))

    return IP_loss, GD_loss, PTD_loss


def stft_consistency_loss(rea_r, rea_g, imag_r, imag_g):
    C_loss = torch.mean(
        torch.mean((rea_r - rea_g) ** 2 + (imag_r - imag_g) ** 2, (1, 2))
    )

    return C_loss


def amp_phase_spectrum(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )  # [batch_size, n_fft//2+1, frames, 2]

    log_amplitude = torch.log(
        stft_spec.abs() + 1e-5
    )  # [batch_size, n_fft//2+1, frames]
    phase = stft_spec.angle()  # [batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec.real, stft_spec.imag


def freev_loss(log, pred, gt_audio, train):
    if pred.log_amplitude is not None:
        gt_log_amplitude, gt_phase, gt_real, gt_imag = amp_phase_spectrum(
            gt_audio,
            train.model_config.n_fft,
            train.model_config.hop_length,
            train.model_config.win_length,
        )
        loss_amplitude = amplitude_loss(gt_log_amplitude, pred.log_amplitude)

        L_IP, L_GD, L_PTD = phase_loss(
            gt_phase,
            pred.phase,
            train.model_config.n_fft,
            pred.phase.size()[-1],
        )
        # Losses defined on phase spectra
        loss_phase = L_IP + L_GD + L_PTD
        _, _, rea_g_final, imag_g_final = amp_phase_spectrum(
            pred.audio.squeeze(1),
            train.model_config.n_fft,
            train.model_config.hop_length,
            train.model_config.win_length,
        )
        loss_consistency = stft_consistency_loss(
            pred.real, rea_g_final, pred.imaginary, imag_g_final
        )
        loss_real_part = F.l1_loss(gt_real, pred.real)
        loss_imaginary_part = F.l1_loss(gt_imag, pred.imaginary)
        loss_stft_reconstruction = loss_consistency + 2.25 * (
            loss_real_part + loss_imaginary_part
        )
        log.add_loss("amplitude", 4.5 * loss_amplitude)
        log.add_loss("phase", 9 * loss_phase)
        log.add_loss("stft_reconstruction", 2 * loss_stft_reconstruction)


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, *, mpd, mrd, msbd):
        super(DiscriminatorLoss, self).__init__()
        self.discriminators = torch.nn.ModuleDict(
            {
                "mpd": DiscriminatorLossHelper(mpd, 5),
                "mrd": DiscriminatorLossHelper(mrd, 3),
                "msbd": DiscriminatorLossHelper(msbd, 3),
            }
        )

    def get_disc_lr_multiplier(self, key):
        return self.discriminators[key].get_disc_lr_multiplier()

    def forward(self, audio_gt, audio, used):
        loss = 0
        for key in used:
            loss += self.discriminators[key](audio_gt, audio)
        return loss.mean()

    def state_dict(self, *args, **kwargs):
        state = {}
        for key, helper in self.discriminators.items():
            state[f"discriminators.{key}.last_loss"] = helper.last_loss
            state[f"discriminators.{key}.weight"] = 1
        return state

    def load_state_dict(self, state_dict, strict=True):
        for key, helper in self.discriminators.items():
            if f"discriminators.{key}.last_loss" in state_dict:
                helper.last_loss = state_dict[f"discriminators.{key}.last_loss"]
        return state_dict


class DiscriminatorLossHelper(torch.nn.Module):
    """
    Discriminator Loss Helper: Returns discriminator loss for a single discriminator
    """

    def __init__(self, model, sub_count):
        super(DiscriminatorLossHelper, self).__init__()
        self.model = model
        self.last_loss = 0.5 * sub_count
        self.ideal_loss = 0.5 * sub_count
        self.f_max = 4.0
        self.h_min = 0.1
        self.x_max = 0.05 * sub_count
        self.x_min = 0.05 * sub_count

    def get_disc_lr_multiplier(self):
        x = abs(self.last_loss - self.ideal_loss)
        result = 1.0
        if self.last_loss > self.ideal_loss:
            result = min(math.pow(self.f_max, x / self.x_max), self.f_max)
        else:
            result = max(math.pow(self.h_min, x / self.x_min), self.h_min)
        return result

    def discriminator_loss(
        self,
        real_score: List[torch.Tensor],
        gen_score: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_score, gen_score):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss

        return loss

    def tprls_loss(
        self,
        real_score: List[torch.Tensor],
        gen_score: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_score, gen_score):
            tau = 0.04
            m_dg = torch.median((dr - dg))
            l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[dr < dg + m_dg])
            loss += tau - F.relu(tau - l_rel)
        return loss

    def forward(self, audio_gt, audio):
        real_score, gen_score, _, _ = self.model(audio_gt, audio)
        disc = self.discriminator_loss(real_score, gen_score)
        tprls = self.tprls_loss(real_score, gen_score)
        self.last_loss = self.last_loss * 0.95 + disc.item() * 0.05
        return disc + tprls


class GeneratorLoss(torch.nn.Module):
    def __init__(self, *, mpd, mrd, msbd):
        super(GeneratorLoss, self).__init__()
        self.generators = torch.nn.ModuleDict(
            {
                "mpd": GeneratorLossHelper(mpd),
                "mrd": GeneratorLossHelper(mrd),
                "msbd": GeneratorLossHelper(msbd),
            }
        )

    def forward(self, audio_gt, audio, used):
        loss = 0
        for key in used:
            loss += self.generators[key](audio_gt, audio)
        return loss.mean()


class GeneratorLossHelper(torch.nn.Module):
    """
    Generator Loss Helper: Returns generator loss for a single discriminator
    """

    def __init__(self, model):
        super(GeneratorLossHelper, self).__init__()
        self.model = model

    def generator_loss(self, gen_score: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for dg in gen_score:
            loss += torch.mean((1 - dg) ** 2)
        return loss

    def feature_loss(
        self, real_features: List[torch.Tensor], gen_features: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_features, gen_features):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    def tprls_loss(self, real_score, gen_score):
        loss = 0
        for dg, dr in zip(real_score, gen_score):
            tau = 0.04
            m_DG = torch.median((dr - dg))
            L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
            loss += tau - F.relu(tau - L_rel)
        return loss

    def forward(self, audio_gt, audio):
        real_score, gen_score, real_features, gen_features = self.model(audio_gt, audio)
        feature = self.feature_loss(real_features, gen_features)
        gen = self.generator_loss(gen_score)
        tprls = self.tprls_loss(real_score, gen_score)
        return feature + gen + tprls


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            wav_tensor = torch.stack(wav_embeddings)
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(1), output_hidden_states=True
        ).hidden_states
        y_rec_tensor = torch.stack(y_rec_embeddings)
        return torch.nn.functional.l1_loss(wav_tensor, y_rec_tensor)


def compute_duration_ce_loss(
    duration_prediction: List[torch.Tensor],
    duration: List[torch.Tensor],
    text_length: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the duration and binary cross-entropy losses over a batch.
    Returns (loss_ce, loss_dur).
    """
    loss_ce = 0
    loss_dur = 0
    for pred, dur, length in zip(duration_prediction, duration, text_length):
        pred = pred[:length, :]
        dur = dur[:length].long()
        target = torch.zeros_like(pred)
        for i in range(target.shape[0]):
            target[i, : dur[i]] = 1
        dur_pred = torch.sigmoid(pred).sum(dim=1)
        loss_dur += F.l1_loss(dur_pred[1 : length - 1], dur[1 : length - 1])
        loss_ce += F.binary_cross_entropy_with_logits(pred.flatten(), target.flatten())
    n = len(text_length)
    return loss_ce / n, loss_dur / n


def duration_loss(*, pred, gt_attn, lengths, mask):
    pred = pred.squeeze(1)
    gt = torch.log(1e-8 + gt_attn.sum(dim=2)) * ~mask
    loss = 0
    for pred_item, gt_item, length_item in zip(pred, gt, lengths):
        loss += torch.sum(
            (gt_item[1 : length_item - 1] - pred_item[1 : length_item - 1]) ** 2
        ) / (length_item - 2)
    return loss / lengths.shape[0]


# The following code was adapated from: https://github.com/huangruizhe/audio/blob/aligner_label_priors/examples/asr/librispeech_alignment/loss.py

# BSD 2-Clause License
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class CTCLossWithLabelPriors(nn.Module):
    def __init__(self, prior_scaling_factor=0.0, blank=0, reduction="mean"):
        super().__init__()

        self.blank = blank
        self.reduction = reduction

        self.log_priors = None
        self.log_priors_sum = None
        self.num_samples = 0
        self.prior_scaling_factor = prior_scaling_factor  # This corresponds to the `alpha` hyper parameter in the paper

    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        # https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py#L181

        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # Be careful: the targets here are already padded! We need to remove paddings from it
        res = targets[indices].tolist()
        res_lengths = target_lengths[indices].tolist()
        res = [l[:l_len] for l, l_len in zip(res, res_lengths)]

        return supervision_segments, res, indices

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        step_type="train",
    ) -> Tensor:
        supervision_segments, token_ids, indices = self.encode_supervisions(
            targets, target_lengths, input_lengths
        )

        decoding_graph = k2.ctc_graph(
            token_ids, modified=False, device=log_probs.device
        )

        # Accumulate label priors for this epoch
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        if step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[: int(le.item())])
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.num_samples += T
            log_batch_priors_sum = torch.logsumexp(
                log_probs_flattened, dim=0, keepdim=True
            )
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

            # Apply the label priors
            if self.log_priors is not None and self.prior_scaling_factor > 0:
                log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        # Compute CTC loss
        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,  # (N, T, C)
            supervision_segments,
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10,
            reduction=self.reduction,
            use_double_scores=True,
            target_lengths=target_lengths,
        )

        return loss

    def on_train_epoch_end(self, train):
        if self.log_priors_sum is not None:
            log_priors_sums = train.accelerator.gather(self.log_priors_sum.unsqueeze(0))
            log_priors_sums = torch.logsumexp(log_priors_sums, dim=0, keepdim=True)
            num_samples = train.accelerator.gather(
                torch.Tensor([self.num_samples]).to(log_priors_sums.device)
            )
            num_samples = num_samples.sum().log().to(log_priors_sums.device)
            new_log_prior = log_priors_sums - num_samples
            if False:
                print(
                    "new_priors: ",
                    ["{0:0.2f}".format(i) for i in new_log_prior[0][0].exp().tolist()],
                )
                print(
                    "new_log_prior: ",
                    ["{0:0.2f}".format(i) for i in new_log_prior[0][0].tolist()],
                )
                if self.log_priors is not None:
                    _a1 = new_log_prior.exp()
                    _b1 = self.log_priors.exp()
                    print(
                        "diff%: ",
                        [
                            "{0:0.2f}".format(i)
                            for i in ((_a1 - _b1) / _b1 * 100)[0][0].tolist()
                        ],
                    )

            prior_threshold = -12.0
            new_log_prior = torch.where(
                new_log_prior < prior_threshold, prior_threshold, new_log_prior
            )

            self.log_priors = new_log_prior
            self.log_priors_sum = None
            self.num_samples = 0
