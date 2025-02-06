import torch
from transformers import AlbertConfig, AlbertModel
import safetensors
from huggingface_hub import hf_hub_download


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(config):
    albert_base_configuration = AlbertConfig(
        vocab_size=config.text_encoder.n_token,
        **(config.plbert.dict()))
    bert = CustomAlbert(albert_base_configuration)
    params = safetensors.torch.load_file(
        hf_hub_download(repo_id="stylish-tts/plbert", filename="plbert.safetensors")
    )
    bert.load_state_dict(params, strict=False)
    return bert
