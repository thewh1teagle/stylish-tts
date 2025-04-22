import click
import logging
from os import path as osp
import pathlib

from einops import rearrange
import numpy
from safetensors.torch import load_file, save_file
import soundfile
import torch
import torchaudio
import tqdm

from stylish_train.models.text_aligner import tdnn_blstm_ctc_model_base
from stylish_train.config_loader import load_config_yaml, load_model_config_yaml
from stylish_train.text_utils import TextCleaner
from stylish_train.utils import get_data_path_list
from stylish_train.meldataset import get_frame_count, get_time_bin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

logger = logging.getLogger(__name__)

to_mel = None


@click.command()
@click.option("-p", "--config_path", default="configs/new.config.yml", type=str)
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--out", type=str)
@click.option("--model", type=str)
def main(config_path, model_config_path, out, model):
    global to_mel
    if osp.exists(config_path):
        config = load_config_yaml(config_path)
    else:
        logger.error(f"Config file not found at {config_path}")
        exit(1)
    if osp.exists(model_config_path):
        model_config = load_model_config_yaml(model_config_path)
    else:
        logger.error(f"Config file not found at {model_config_path}")
        exit(1)

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80,  # align seems to perform worse on higher n_mels
        n_fft=model_config.n_fft,
        win_length=model_config.win_length,
        hop_length=model_config.hop_length,
        sample_rate=model_config.sample_rate,
    )

    aligner_dict = load_file(model, device=device)
    aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.n_token
    )
    aligner = aligner.to(device)
    aligner.load_state_dict(aligner_dict)

    text_cleaner = TextCleaner(model_config.symbol)

    wavdir = pathlib.Path(config.dataset.wav_path)
    vals = calculate_alignments(
        pathlib.Path(config.dataset.val_data),
        wavdir,
        aligner,
        model_config,
        text_cleaner,
    )
    trains = calculate_alignments(
        pathlib.Path(config.dataset.train_data),
        wavdir,
        aligner,
        model_config,
        text_cleaner,
    )
    result = vals | trains
    save_file(result, out)


def preprocess(wave):
    mean, std = -4, 4
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    mel_tensor = mel_tensor[:, :, :-1]
    return mel_tensor


@torch.no_grad()
def calculate_alignments(path, wavdir, aligner, model_config, text_cleaner):
    alignment_map = {}
    scores_map = {}
    iterator = tqdm.tqdm(
        iterable=audio_list(path, wavdir, model_config),
        desc="Aligning",
        unit="segments",
        initial=0,
        colour="MAGENTA",
        dynamic_ncols=True,
    )
    for name, text_raw, wave in iterator:
        mels = preprocess(wave).to(device)
        text = text_cleaner("$" + text_raw + "$")
        text = torch.tensor(text).to(device).unsqueeze(0)
        mels = rearrange(mels, "b f t -> b t f")
        mel_lengths = torch.zeros([1], dtype=int, device=device)
        mel_lengths[0] = mels.shape[1]
        prediction, _ = aligner(mels, mel_lengths)
        prediction = rearrange(prediction, "t b k -> b t k")

        text_lengths = torch.zeros([1], dtype=int, device=device)
        text_lengths[0] = text.shape[1]
        blank = model_config.text_encoder.n_token
        alignment, scores = torchaudio.functional.forced_align(
            log_probs=prediction,
            targets=text,
            input_lengths=mel_lengths // 2,
            target_lengths=text_lengths,
            blank=blank,
        )
        alignment = alignment.squeeze()
        atensor = torch.zeros(
            [1, text.shape[1], alignment.shape[0]], device=mels.device, dtype=bool
        )
        text_index = 0
        last_text = alignment[0]
        was_blank = False
        for i in range(alignment.shape[0]):
            if alignment[i] == blank:
                was_blank = True
            else:
                if alignment[i] != last_text or was_blank:
                    text_index += 1
                    last_text = alignment[i]
                    was_blank = False
            assert alignment[i] == blank or alignment[i] == text[0, text_index]
            atensor[0, text_index, i] = 1
        alignment_map[name] = atensor
        scores_map[name] = scores.exp().mean().item()
    with open("scores.txt", "w") as f:
        for name in scores_map.keys():
            f.write(str(scores_map[name]) + " " + name + "\n")
    return alignment_map


def audio_list(path, wavdir, model_config):
    with path.open("r") as f:
        for line in f:
            fields = line.split("|")
            name = fields[0]
            phonemes = fields[1]
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                sys.stderr.write(f"Skipping {name}: Wrong sample rate ({sr})")
            if wave.shape[-1] == 2:
                wave = wave[:, 0].squeeze()
            time_bin = get_time_bin(wave.shape[0], model_config.hop_length)
            if time_bin == -1:
                sys.stderr.write(f"Skipping {name}: Too short\n")
                continue
            frame_count = get_frame_count(time_bin)
            pad_start = (frame_count * 300 - wave.shape[0]) // 2
            pad_end = frame_count * 300 - wave.shape[0] - pad_start
            wave = numpy.concatenate(
                [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
            )
            yield name, phonemes, wave


if __name__ == "__main__":
    main()
