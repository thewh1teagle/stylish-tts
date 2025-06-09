"""
uv pip install espeakng-loader phonemizer-fork
uv run hebrew/phonemize.py
"""

from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import pandas as pd
from pathlib import Path
import re

# Initialize espeak-ng backend paths explicitly
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

def remove_space_around_punctuation(text):
    return re.sub(r'\s*([.,?!;:])\s*', r'\1', text)

def phonemize_text(txt):
    return phonemize(
        txt,
        backend='espeak',
        language='en-us',
        strip=True,
        with_stress=True,
        preserve_punctuation=True
    )

def main():
    src_folder = Path('./24khz_dataset')
    src_metadata_path = src_folder / 'metadata_text.csv'
    target_metadata_path = src_folder / 'metadata.csv'

    df = pd.read_csv(src_metadata_path, sep='|', header=None, names=['id', 'text'])
    
    # Clean original text
    df['text'] = df['text'].apply(remove_space_around_punctuation)

    # Using apply to phonemize all texts
    df['phonemes'] = df['text'].apply(phonemize_text)

    # Clean phonemes output
    df['phonemes'] = df['phonemes'].apply(remove_space_around_punctuation)

    df['speaker_id'] = 0
    df['filename'] = df['id'].astype(str) + '.wav'  # add extension here if missing

    output_df = df[['filename', 'phonemes', 'speaker_id', 'text']]

    output_df.to_csv(target_metadata_path, sep='|', index=False, header=False)

if __name__ == '__main__':
    main()
