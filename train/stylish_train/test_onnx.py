import torch
from stylish_lib.config_loader import ModelConfig
from stylish_lib.text_utils import TextCleaner
import torch
import numpy as np
import torch
import onnxruntime as ort
import click
from scipy.io.wavfile import write
import onnx


def read_meta_data_onnx(filename, key):
    model = onnx.load(filename)
    for prop in model.metadata_props:
        if prop.key == key:
            return prop.value
    return None


@click.command()
@click.option("--dir", type=str)
@click.option("--text", type=str, multiple=True, help="A list of phonemes")
def main(dir, text):
    texts = text
    model_config = read_meta_data_onnx(f"{dir}/stylish.onnx", "model_config")
    assert (
        model_config
    ), "model_config metadata not found. Please rerun ONNX conversion."
    model_config = ModelConfig.model_validate_json(model_config)
    text_cleaner = TextCleaner(model_config.symbol)
    session = ort.InferenceSession(
        f"{dir}/stylish.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    for i, text in enumerate(texts):
        tokens = torch.tensor(text_cleaner(text)).unsqueeze(0)
        texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int)
        texts[0][1 : tokens.shape[1] + 1] = tokens
        text_lengths = torch.zeros([1], dtype=int)
        text_lengths[0] = tokens.shape[1] + 2
        text_mask = torch.zeros(1, texts.shape[1], dtype=bool)
        # Load ONNX model

        outputs = session.run(
            None,
            {
                "texts": texts.cpu().numpy(),
                "text_lengths": text_lengths.cpu().numpy(),
            },
        )
        outfile = f"{dir}/sample_{i}.wav"
        print("Saving to:", outfile)
        combined = np.multiply(outputs[0], 32768)
        write(outfile, 24000, combined.astype(np.int16))


if __name__ == "__main__":
    main()
