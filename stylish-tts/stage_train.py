import math
import torch
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log
from losses import magphase_loss, compute_duration_ce_loss, freev_loss


def train_pre_acoustic(batch, model, train) -> LossLog:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch, use_random_mono=True)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
    freev_loss(log, pred, batch.audio_gt, train)
    train.accelerator.backward(log.backwards_loss() * math.sqrt(batch.text.shape[0]))
    return log.detach()


def train_acoustic(batch, model, train) -> LossLog:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch)
        train.stage.optimizer.zero_grad()
        d_loss = train.discriminator_loss(
            batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        ).mean()
        train.accelerator.backward(d_loss)
        train.stage.optimizer.step("msd")
        train.stage.optimizer.step("mpd")
        train.stage.optimizer.zero_grad()

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

        loss_s2s = 0
        for pred_align, text, length in zip(
            state.s2s_pred, batch.text, batch.text_length
        ):
            loss_s2s += torch.nn.functional.cross_entropy(
                pred_align[:length], text[:length]
            )
        loss_s2s /= batch.text.size(0)
        log.add_loss("s2s", loss_s2s)

        log.add_loss(
            "mono", torch.nn.functional.l1_loss(*(state.duration_results)) * 10
        )

        freev_loss(log, pred, batch.audio_gt, train)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )
        log.add_loss("discriminator", d_loss)

    return log.detach()


def train_pre_textual(batch, model, train) -> LossLog:
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

    return log.detach()


def train_textual(batch, model, train) -> LossLog:
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

    return log.detach()


def train_joint(batch, model, train) -> LossLog:
    state = BatchContext(train=train, model=model, text_length=batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        d_loss = train.discriminator_loss(
            batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        ).mean()
        train.accelerator.backward(d_loss)
        train.stage.optimizer.step("msd")
        train.stage.optimizer.step("mpd")
        train.stage.optimizer.zero_grad()
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
        log.add_loss("discriminator", d_loss)

    return log.detach()
