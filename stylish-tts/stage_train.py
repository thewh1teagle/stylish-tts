from batch_context import BatchContext
from loss_log import LossLog, build_loss_log, combine_logs


def train_pre_acoustic(batch, model, train) -> LossLog:
    split_count = 1
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        decoding = state.acoustic_prediction(batch, split=split_count)
        train.stage.optimizer.zero_grad()
        loglist = []
        for audio_out, audio_gt_slice, _, _ in decoding:
            log = build_loss_log(train)
            log.add_loss(
                "mel",
                train.stft_loss(audio_out.squeeze(1), audio_gt_slice) / split_count,
            )
            train.accelerator.backward(log.total(), retain_graph=True)
            loglist.append(log)
    return combine_logs(loglist).detach()
