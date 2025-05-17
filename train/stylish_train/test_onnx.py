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
from models.onnx_models import Stylish, CustomSTFT
import torch
import torch.nn as nn


from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import click
from scipy.io.wavfile import write


@click.command()
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--dir", type=str)
@click.option("--checkpoint", default="", type=str)
def main(model_config_path, dir, checkpoint):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model_config = load_model_config_yaml(model_config_path)
    text_cleaner = TextCleaner(model_config.symbol)
    sbert = SentenceTransformer(model_config.sbert.model).cpu()

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
    texts[0][1 : tokens.shape[1] + 1] = tokens
    text_lengths = torch.zeros([1], dtype=int).to(device)
    text_lengths[0] = tokens.shape[1] + 2
    text_mask = torch.zeros(1, texts.shape[1], dtype=bool).to(device)
    sentence_embedding = (
        torch.from_numpy(
            sbert.encode(
                [
                    # "These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come."
                    "Toto, of course, slept beside his little mistress."
                ],
                show_progress_bar=False,
            )
        )
        .float()
        .to(device)
    )
    # Load ONNX model
    session = ort.InferenceSession(
        f"{dir}/stylish.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    outputs = session.run(
        None,
        {
            "texts": texts.cpu().numpy(),
            "text_mask": text_mask.cpu().numpy(),
            "sentence_embedding": sentence_embedding.cpu().numpy(),
        },
    )
    outfile = f"{dir}/sample.wav"
    print("Saving to:", outfile)
    combined = np.multiply(outputs[0], 32768)
    write(outfile, 24000, combined.astype(np.int16))

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
