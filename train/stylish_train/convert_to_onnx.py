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
from models.onnx_models import Stylish, Generator, CustomSTFT
import click

from attr import attr
import numpy as np


@click.command()
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--out_dir", type=str)
@click.option("--checkpoint", default="", type=str)
def main(model_config_path, out_dir, checkpoint):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    if osp.exists(model_config_path):
        model_config = load_model_config_yaml(model_config_path)
    else:
        exit(f"Model configuration not found: {model_config_path}")
    if out_dir is None:
        exit(f"No out_dir was specified.")
    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not osp.exists(out_dir):
        exit(f"Failed to create or find out_dir at {out_dir}.")

    text_cleaner = TextCleaner(model_config.symbol)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()
    model = Stylish(model_config, device).eval()
    stft = CustomSTFT(
        filter_length=model.model.generator.gen_istft_n_fft,
        hop_length=model.model.generator.gen_istft_hop_size,
        win_length=model.model.generator.gen_istft_n_fft,
    )
    model.model.generator.stft = stft.to(device).eval()
    generator = Generator(model.model.generator)

    texts = (
        torch.tensor(text_cleaner("ɑɐɒæɓʙβɔɗɖðʤəɘɚɛɜɝɞɟʄɡɠ")).unsqueeze(0).to(device)
    )
    text_lengths = torch.zeros([1], dtype=int).to(device)
    text_lengths[0] = texts.shape[1]
    text_mask = torch.ones(1, texts.shape[1], dtype=bool).to(device)
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
        .to(device)
    )

    inputs = (texts, text_lengths, text_mask, sentence_embedding)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            opset_version=14,
            f=f"{out_dir}/stylish.onnx",
            input_names=["texts", "text_lengths", "text_mask", "sentence_embedding"],
            output_names=["waveform"],
            dynamic_axes={
                "texts": {1: "num_token"},
                "text_mask": {1: "num_token"},
                "waveform": {0: "num_samples"},
            },
        )

    # input_shapes = (
    #     torch.Size([1, 512, 1150]),
    #     torch.Size([1, 128]),
    #     torch.Size([1, 1150]),
    #     torch.Size([1, 1150]),
    # )
    # input_dtypes = torch.float32, torch.float32, torch.float32, torch.float32
    # input_names = "mel, style, pitch, energy".split(", ")
    # with torch.no_grad():
    #     torch.onnx.export(
    #         generator,
    #         tuple(
    #             [
    #                 torch.ones(input_shape, dtype=dtype).to(device)
    #                 for input_shape, dtype in zip(input_shapes, input_dtypes)
    #             ]
    #         ),
    #         opset_version=19,
    #         f=f"{out_dir}/ringformer.onnx",
    #         input_names=input_names,
    #         output_names=["waveform"],
    #         dynamic_axes=dict(
    #             {
    #                 k: {
    #                     i: f"dim_{i}"
    #                     for i, d in enumerate(v)
    #                     if d > 1 and d != 512 and d != 640
    #                 }
    #                 for k, v in zip(input_names, input_shapes)
    #             }
    #         ),
    #     )


if __name__ == "__main__":
    main()
