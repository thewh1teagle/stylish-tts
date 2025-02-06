import os
import yaml
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


def load_plbert(checkpoint_path, config_path):
    plbert_config = yaml.safe_load(open(config_path))

    albert_base_configuration = AlbertConfig(**plbert_config["model_params"])
    bert = CustomAlbert(albert_base_configuration)

    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # state_dict = checkpoint["net"]
    # from collections import OrderedDict

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #    name = k[7:]  # remove `module.`
    #    if name.startswith("encoder."):
    #        name = name[8:]  # remove `encoder.`
    #        new_state_dict[name] = v
    # del new_state_dict["embeddings.position_ids"]
    new_state_dict = safetensors.torch.load_file(
        hf_hub_download(repo_id="stylish-tts/plbert", filename="plbert.safetensors")
    )
    bert.load_state_dict(new_state_dict, strict=False)

    return bert
