import random
import torch
import torchaudio
from torch.nn import functional as F
from einops import rearrange

from batch_context import BatchContext
from loss_log import build_loss_log
from losses import compute_duration_ce_loss, duration_loss
from utils import length_to_mask


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length // 2, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.n_token
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=batch.mel_length[i].unsqueeze(0) // 2,
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


@torch.no_grad()
def validate_acoustic(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    # log.add_loss(
    #     "pitch",
    #     torch.nn.functional.smooth_l1_loss(state.calculate_pitch(batch), state.pitch_prediction),
    # )
    # log.add_loss(
    #     "energy",
    #     torch.nn.functional.smooth_l1_loss(state.acoustic_energy(batch.mel), state.energy_prediction),
    # )
    # log.add_loss("duration", duration_loss(pred=state.duration_prediction, gt_attn=state.duration_results[1], lengths=batch.text_length, mask=state.text_mask))
    return log, state.get_attention(), pred.audio, batch.audio_gt


@torch.no_grad()
def validate_textual(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.textual_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, state.pitch_prediction),
    )
    log.add_loss(
        "energy", torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction)
    )
    # log.add_loss(
    #     "duration",
    #     duration_loss(
    #         pred=state.duration_prediction,
    #         gt_attn=state.duration_results[1],
    #         lengths=batch.text_length,
    #         mask=state.text_mask,
    #     ),
    # )
    loss_ce, loss_dur = compute_duration_ce_loss(
        state.duration_prediction,
        state.duration_results[1].sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    return log, state.get_attention(), pred.audio, batch.audio_gt


@torch.no_grad()
def validate_sbert(batch, train):
    """Validation function for the sbert stage."""
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.sbert_prediction_single(batch)
    # 1. Get textual and acoustic embeddings
    textual_style_embedding = state.textual_style_embedding(batch.sentence_embedding)
    textual_prosody_embedding = state.textual_prosody_embedding(
        batch.sentence_embedding
    )
    acoustic_style_embedding = state.acoustic_style_embedding(batch.mel)
    acoustic_prosody_embedding = state.acoustic_prosody_embedding(batch.mel)

    log = build_loss_log(train)

    # 2. Calculate Loss
    style_loss = torch.nn.functional.l1_loss(
        textual_style_embedding, acoustic_style_embedding
    )
    prosody_loss = torch.nn.functional.l1_loss(
        textual_prosody_embedding, acoustic_prosody_embedding
    )

    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log))
    log.add_loss("sbert_style_loss", style_loss)
    log.add_loss("sbert_prosody_loss", prosody_loss)
    return log, state.get_attention(), pred.audio, batch.audio_gt
