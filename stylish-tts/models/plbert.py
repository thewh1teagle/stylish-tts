"""
Pretrained PLBERT

 - Phoneme level BERT using the ALBERT architecture but trained on phonemes instead of words
 - Taken from StyleTTS 2 repo: https://github.com/yl4579/StyleTTS2
 - Paper: https://arxiv.org/abs/2301.08810
 - Training Code: https://github.com/yl4579/PL-BERT
"""

import transformers

class PLBERT(transformers.AlbertModel):
    def __init__(self, **kwargs):
        super().__init__(transformers.AlbertConfig(**kwargs))

    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state
