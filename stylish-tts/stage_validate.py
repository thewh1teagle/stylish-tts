import torch

from batch_context import BatchContext
from loss_log import build_loss_log
from losses import compute_duration_ce_loss


@torch.no_grad()
def validate_acoustic(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), batch.audio_gt))
    return log, state.get_attention(), pred.audio[0], batch.audio_gt[0]


@torch.no_grad()
def validate_textual(batch, train):
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.textual_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), batch.audio_gt))
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, state.pitch_prediction),
    )
    log.add_loss(
        "energy", torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction)
    )
    loss_ce, loss_dur = compute_duration_ce_loss(
        state.duration_prediction,
        state.duration_results[1].sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    return log, state.get_attention(), pred.audio[0], batch.audio_gt[0]


@torch.no_grad()
def validate_sbert(batch, train):
    """Validation function for the sbert stage."""
    state = BatchContext(train=train, model=train.model, text_length=batch.text_length)
    pred = state.sbert_prediction_single(batch)
    # 1. Get textual and acoustic embeddings
    textual_style_embedding = state.textual_style_embedding(batch.sentence_embedding)
    textual_prosody_embedding = state.textual_prosody_embedding(batch.sentence_embedding)
    acoustic_style_embedding = state.acoustic_style_embedding(batch.mel)
    acoustic_prosody_embedding = state.acoustic_prosody_embedding(batch.mel)

    log = build_loss_log(train)

    # 2. Calculate Loss
    style_loss = torch.nn.functional.l1_loss(textual_style_embedding, acoustic_style_embedding)
    prosody_loss = torch.nn.functional.l1_loss(textual_prosody_embedding, acoustic_prosody_embedding)
    
    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), batch.audio_gt))
    log.add_loss("sbert_style_loss", style_loss)
    log.add_loss("sbert_prosody_loss", prosody_loss)
    return log, state.get_attention(), pred.audio[0], batch.audio_gt[0] 
