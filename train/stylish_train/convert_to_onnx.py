import os.path as osp

import click
import onnx
import torch
from sentence_transformers import SentenceTransformer

from models.models import build_model
from models.onnx_models import Stylish, CustomSTFT
from config_loader import load_model_config_yaml
from text_utils import TextCleaner


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
    text_cleaner = TextCleaner(model_config.symbol)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()
    modules = build_model(model_config)
    model = Stylish(**modules, device=device).eval()
    stft = CustomSTFT(
        filter_length=model.generator.gen_istft_n_fft,
        hop_length=model.generator.gen_istft_hop_size,
        win_length=model.generator.gen_istft_n_fft,
    )
    model.generator.stft = stft.to(device).eval()
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
            f="stylish.onnx",
            input_names=["texts", "text_lengths", "text_mask", "sentence_embedding"],
            output_names=["waveform"],
            dynamic_axes={
                "texts": {1: "num_token"},
                "text_mask": {1: "num_token"},
                "waveform": {0: "num_samples"},
            },
            verify=True,
        )

    onnx_model = onnx.load("stylish.onnx")
    for node in onnx_model.graph.node:
        if node.op_type == "Transpose":
            if node.name == "/text_encoder_1/Transpose_7":
                perm = list(node.attribute[0].ints)
                perm = [2 if i == -1 else i for i in perm]
                node.attribute[0].ints[:] = perm
    onnx.save(onnx_model, "stylish.onnx")
    print("Exported!")


if __name__ == "__main__":
    main()
