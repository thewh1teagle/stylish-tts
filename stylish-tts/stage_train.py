import torch
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log, combine_logs
from losses import freev_loss


def train_pre_acoustic(batch, model, train) -> LossLog:
    split_count = 1
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        decoding = state.acoustic_prediction(batch, split=split_count)
        train.stage.optimizer.zero_grad()
        loglist = []
        for pred, audio_gt_slice, _, _ in decoding:
            log = build_loss_log(train)
            log.add_loss(
                "mel",
                train.stft_loss(pred.audio.squeeze(1), audio_gt_slice) / split_count,
            )
            train.accelerator.backward(log.total(), retain_graph=True)
            loglist.append(log)
    return combine_logs(loglist).detach()


def train_acoustic(batch, model, train) -> LossLog:
    split_count = 1
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        decoding = state.acoustic_prediction(batch, split=split_count)
        train.stage.optimizer.zero_grad()
        loglist = []
        for pred, audio_gt_slice, begin, end in decoding:
            log = build_loss_log(train)
            log.add_loss(
                "mel",
                train.stft_loss(pred.audio.squeeze(1), audio_gt_slice) / split_count,
            )
            log.add_loss(
                "gen",
                train.generator_loss(
                    audio_gt_slice.detach().unsqueeze(1).float(), pred.audio
                ).mean(),
            )
            log.add_loss("slm", train.wavlm_loss(audio_gt_slice.detach(), pred.audio))
            freev_loss(log, batch, pred, begin, end, audio_gt_slice, train)
            train.accelerator.backward(log.total(), retain_graph=True)
            d_loss = train.discriminator_loss(
                audio_gt_slice.detach().unsqueeze(1).float(), pred.audio.detach()
            ).mean()
            train.accelerator.backward(d_loss, retain_graph=True)
            log.add_loss("discriminator", d_loss)
            loglist.append(log)
        train.stage.optimizer.step("msd")
        train.stage.optimizer.step("mpd")

    incremental_log = combine_logs(loglist).detach()
    log = build_loss_log(train)
    loss_s2s = 0
    for pred, text, length in zip(state.s2s_pred, batch.text, batch.text_length):
        loss_s2s += torch.nn.functional.cross_entropy(pred[:length], text[:length])
    loss_s2s /= batch.text.size(0)
    log.add_loss("s2s", loss_s2s)
    log.add_loss("mono", torch.nn.functional.l1_loss(*(state.duration_results)) * 10)
    train.accelerator.backward(log.total())

    return combine_logs([incremental_log, log]).detach()
