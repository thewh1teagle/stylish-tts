import torch

from batch_context import BatchContext
from loss_log import LossLog, build_loss_log, combine_logs


@torch.no_grad()
def validate_acoustic(batch, train) -> LossLog:
    state = BatchContext(train, train.model, batch.text_length)
    decoding = state.acoustic_prediction(batch, split=1)
    first_decoding = None
    for item in decoding:
        first_decoding = item
        break
    pred, audio_gt_slice, _, _ = first_decoding
    log = build_loss_log(train)
    log.add_loss("mel", train.stft_loss(pred.audio.squeeze(1), audio_gt_slice))
    return log, state.get_attention(), pred.audio[0], audio_gt_slice[0]
