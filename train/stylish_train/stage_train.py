import math
import random
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
from einops import rearrange
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log
from losses import magphase_loss, compute_duration_ce_loss, freev_loss, duration_loss
from utils import length_to_mask


def train_alignment(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length // 2, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss() * math.sqrt(batch.text.shape[0]))
    return log.detach(), None


def train_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch)
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(),
                pred.audio,
            ).mean(),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )

        freev_loss(log, pred, batch.audio_gt, train)
        # log.add_loss(
        #     "pitch",
        #     torch.nn.functional.smooth_l1_loss(state.calculate_pitch(batch), state.pitch_prediction),
        # )
        # log.add_loss(
        #     "energy",
        #     torch.nn.functional.smooth_l1_loss(state.acoustic_energy(batch.mel), state.energy_prediction),
        # )
        # log.add_loss("duration", duration_loss(pred=state.duration_prediction, gt_attn=state.duration_results[1], lengths=batch.text_length, mask=state.text_mask))
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), pred.audio.detach()


def train_pre_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        state.textual_bootstrap_prediction(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        log.add_loss(
            "duration",
            duration_loss(
                pred=state.duration_prediction,
                gt_attn=state.duration_results[1],
                lengths=batch.text_length,
                mask=state.text_mask,
            ),
        )
        # loss_ce, loss_dur = compute_duration_ce_loss(
        #     state.duration_prediction,
        #     state.duration_results[1].sum(dim=-1),
        #     batch.text_length,
        # )
        # log.add_loss("duration_ce", loss_ce)
        # log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), None


def train_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        log.add_loss(
            "duration",
            duration_loss(
                pred=state.duration_prediction,
                gt_attn=state.duration_results[1],
                lengths=batch.text_length,
                mask=state.text_mask,
            ),
        )
        # loss_ce, loss_dur = compute_duration_ce_loss(
        #     state.duration_prediction,
        #     state.duration_results[1].sum(dim=-1),
        #     batch.text_length,
        # )
        # log.add_loss("duration_ce", loss_ce)
        # log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), pred.audio.detach()


def train_joint(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio
            ).mean(),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(pitch, state.pitch_prediction),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction),
        )
        loss_ce, loss_dur = compute_duration_ce_loss(
            state.duration_prediction,
            state.duration_results[1].sum(dim=-1),
            batch.text_length,
        )
        log.add_loss("duration_ce", loss_ce)
        log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), pred.audio.detach()


def train_sbert(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Training function for the sbert stage."""
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        # 1. Get textual and acoustic embeddings
        textual_style_embedding = state.textual_style_embedding(
            batch.sentence_embedding
        )
        textual_prosody_embedding = state.textual_prosody_embedding(
            batch.sentence_embedding
        )
        acoustic_style_embedding = state.acoustic_style_embedding(batch.mel)
        acoustic_prosody_embedding = state.acoustic_prosody_embedding(batch.mel)

        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)

        # 2. Calculate Loss
        style_loss = torch.nn.functional.l1_loss(
            textual_style_embedding, acoustic_style_embedding
        )
        prosody_loss = torch.nn.functional.l1_loss(
            textual_prosody_embedding, acoustic_prosody_embedding
        )

        log.add_loss("sbert_style_loss", style_loss)
        log.add_loss("sbert_prosody_loss", prosody_loss)

        train.accelerator.backward(log.total())

    return log.detach(), None
