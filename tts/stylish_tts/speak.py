import os.path as osp
import click
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from stylish_lib.config_loader import load_model_config_yaml
from stylish_lib.text_utils import TextCleaner


@click.command()
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("-m", "--model_path", default="", type=str)
def main(model_config_path, model_path):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    if osp.exists(model_config_path):
        model_config = load_model_config_yaml(model_config_path)
    text_cleaner = TextCleaner(model_config.symbol)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()

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

    # Load ONNX model
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    outputs = session.run(
        None,
        {
            "texts": texts.cpu().numpy(),
            "text_lengths": text_lengths.cpu().numpy(),
            "text_mask": text_mask.cpu().numpy(),
            "sentence_embedding": sentence_embedding.cpu().numpy(),
        },
    )
    print("Model output:", outputs[0].shape)


if __name__ == "__main__":
    main()
