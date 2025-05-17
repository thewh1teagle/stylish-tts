import os
import os.path as osp
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce

from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, maximum_path
from models.models import build_model
from stylish_lib.config_loader import load_model_config_yaml
from stylish_lib.text_utils import TextCleaner
from sentence_transformers import SentenceTransformer
from models.onnx_models import Stylish, CustomSTFT

from attr import attr
import numpy as np
from scipy.io.wavfile import write
from utils import length_to_mask


def convert_to_onnx(model_config, out_dir, model_in, device):
    text_cleaner = TextCleaner(model_config.symbol)
    sbert = SentenceTransformer(model_config.sbert.model).cpu()
    model = Stylish(**model_in, device=device).eval()
    stft = CustomSTFT(
        filter_length=model.generator.gen_istft_n_fft,
        hop_length=model.generator.gen_istft_hop_size,
        win_length=model.generator.gen_istft_n_fft,
    )
    model.generator.stft = stft.to(device).eval()

    tokens = (
        torch.tensor(
            text_cleaner(
                "ðˈiːz wˈɜː tˈuː hˈæv ˈæn ɪnˈɔːɹməs ˈɪmpækt , nˈɑːt ˈoʊnliː bɪkˈɔz ðˈeɪ wˈɜː əsˈoʊsiːˌeɪtᵻd wˈɪð kˈɑːnstəntˌiːn , bˈʌt ˈɔlsoʊ bɪkˈɔz , ˈæz ɪn sˈoʊ mˈɛniː ˈʌðɚ ˈɛɹiːəz , ðə dɪsˈɪʒənz tˈeɪkən bˈaɪ kˈɑːnstəntˌiːn ( ˈɔːɹ ɪn hˈɪz nˈeɪm ) wˈɜː tˈuː hˈæv ɡɹˈeɪt səɡnˈɪfɪkəns fˈɔːɹ sˈɛntʃɚiːz tˈuː kˈʌm ."
            )
        )
        .unsqueeze(0)
        .to(device)
    )
    texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int).to(device)
    texts[0, 1 : tokens.shape[1] + 1] = tokens
    text_lengths = torch.zeros([1], dtype=int).to(device)
    text_lengths[0] = tokens.shape[1] + 2
    text_mask = length_to_mask(text_lengths)
    sentence_embedding = (
        torch.from_numpy(
            sbert.encode(
                [
                    # "These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come."
                    "Toto, of course, slept beside his little mistress."
                    # "Once upon a time, a princess lived in a castle."
                ],
                show_progress_bar=False,
            )
        )
        .float()
        .to(device)
    )

    filename = f"{out_dir}/stylish.onnx"
    inputs = (texts, text_mask, sentence_embedding)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            opset_version=14,
            f=filename,
            input_names=["texts", "text_mask", "sentence_embedding"],
            output_names=["waveform"],
            dynamic_axes={
                "texts": {1: "num_token"},
                "text_mask": {1: "num_token"},
                "waveform": {0: "num_samples"},
            },
        )

    # audio = model.forward(texts, text_lengths, text_mask, sentence_embedding)
    # train.writer.add_audio("onnx/sample", audio, 0, sample_rate=24000)
    # outfile = f"{out_dir}/sample-torch.wav"
    # print("Saving to:", outfile)
    # combined = np.multiply(audio.cpu().numpy(), 32768)
    # write(outfile, 24000, combined.astype(np.int16))

    return filename
