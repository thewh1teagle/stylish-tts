import onnxruntime as ort
import numpy as np
from stylish_lib.text_utils import TextCleaner
from stylish_lib.config_loader import load_model_config_yaml
from sentence_transformers import SentenceTransformer
import torch

model_config = load_model_config_yaml("/content/stylish-tts/config/model.yml")
text_cleaner = TextCleaner(model_config.symbol)
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()
texts = torch.tensor(text_cleaner("ɑɐɒæɓʙβɔɗɖðʤəɘɚɛɜɝɞɟʄɡɠ")).unsqueeze(0).cuda()
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
    "stylish.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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
print("Model output:", outputs[0])
