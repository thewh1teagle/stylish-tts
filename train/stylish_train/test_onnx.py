import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce

# import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, maximum_path
from models.models import build_model
from stylish_lib.config_loader import load_model_config_yaml
from stylish_lib.text_utils import TextCleaner
from sentence_transformers import SentenceTransformer
from models.onnx_models import Stylish, Generator, CustomSTFT
import torch
import torch.nn as nn


from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import click


@click.command()
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--dir", type=str)
@click.option("--checkpoint", default="", type=str)
def main(model_config_path, dir, checkpoint):
    model_config = load_model_config_yaml(model_config_path)
    text_cleaner = TextCleaner(model_config.symbol)
    model = Stylish(model_config, "cuda").eval()
    model.model.generator.stft = CustomSTFT(
        filter_length=model.model.generator.gen_istft_n_fft,
        hop_length=model.model.generator.gen_istft_hop_size,
        win_length=model.model.generator.gen_istft_n_fft,
    )
    model.model.generator.stft.cuda().eval()
    gen = Generator(model.model.generator)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()

    texts = (
        torch.tensor(text_cleaner("ɑɐɒæɓʙβɔɗɖðʤəɘɚɛɜɝɞɟʄɡɠɑɐɒæɓʙβɔɗɖðʤəɘɚɛɜɝɞɟʄɡɠ"))
        .unsqueeze(0)
        .cuda()
    )
    text_lengths = torch.zeros([1], dtype=int).cuda()
    text_lengths[0] = texts.shape[1]
    text_mask = torch.ones(1, texts.shape[1], dtype=bool).cuda()
    sentence_embedding = (
        torch.from_numpy(
            sbert.encode(
                [
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
                ],
                show_progress_bar=False,
            )
        )
        .float()
        .cuda()
    )
    # Load ONNX model
    session = ort.InferenceSession(
        f"{dir}/stylish.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print("before")
    outputs = session.run(
        None,
        {
            "texts": texts.cpu().numpy(),
            "text_lengths": text_lengths.cpu().numpy(),
            "text_mask": text_mask.cpu().numpy(),
            "sentence_embedding": sentence_embedding.cpu().numpy(),
        },
    )
    print("after")
    print(outputs)

    # print("Using RingFormer in PyTorch...")
    # torch_outputs = [torch.from_numpy(out).cuda() for out in outputs]
    # print(gen(*torch_outputs))

    # print("Using broken RingFormer in ONNX...")
    # input_names = "mel, style, pitch, energy".split(", ")
    # inp = {name: output for name, output in zip(input_names[:3], outputs[:3])}
    # session = ort.InferenceSession(
    #     f"{dir}/ringformer.onnx",
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )
    # outputs = session.run(None, inp)
    # print(outputs)


if __name__ == "__main__":
    main()
