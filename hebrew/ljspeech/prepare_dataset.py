import pandas as pd
import argparse
from pathlib import Path
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import librosa
from tqdm.auto import tqdm
import re
import pydub

def phonemize_text(txt):
    if not isinstance(txt, str):
        breakpoint()
    return phonemize(
        txt,
        backend='espeak',
        language='en-us',
        strip=True,
        with_stress=True,
        preserve_punctuation=True
    )
    
def normalize_and_resample_wav(src_path, dst_path, target_sr=24000):
    audio = pydub.AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)  # Resample & mono
    normalized_audio = audio.normalize()  # Normalize volume
    normalized_audio.export(dst_path, format="wav")


def get_duration(path):
    return librosa.get_duration(filename=path)

def remove_space_around_punctuation(text):
    return re.sub(r'\s*([.,?!;:])\s*', r'\1', text)

def main(input_folder, output_folder, max_dur=10, required_duration=36000):
    tqdm.pandas()
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    src_metadata_path = input_folder / 'metadata.csv'
    src_wav_path = (input_folder / 'wavs').absolute()
    dst_metadata_path = output_folder / 'metadata.csv'
    dst_wav_path = output_folder / 'wav'
    dst_wav_path.mkdir(exist_ok=True, parents=True)

    src_df = pd.read_csv(
        src_metadata_path,
        sep='|',
        index_col=0,
        names=['file_id', 'text', 'normalized_text'],
        header=None,
        quotechar='"'
    )

    # Drop rows where normalized_text is NaN
    nan_rows = src_df[src_df['normalized_text'].isna()]
    print(f"Found {len(nan_rows)} rows with NaN in 'normalized_text' before filtering:")
    print(nan_rows)

    src_df = src_df.dropna(subset=['normalized_text'])
    print(f"After dropping NaN normalized_text, {len(src_df)} segments remain.")

    # Compute durations and filter by max duration
    src_df['duration'] = src_df.index.map(lambda fid: get_duration(src_wav_path / f"{fid}.wav"))
    src_df = src_df[src_df['duration'] <= max_dur].sort_values(by='duration', ascending=False)

    print(f"Total segments before duration filtering: {len(src_df)}, total duration: {src_df['duration'].sum() / 3600:.2f} hours")

    # Filter by cumulative duration
    cumulative_duration = src_df['duration'].cumsum()
    src_df = src_df[cumulative_duration <= required_duration]
    print(f"Selected {len(src_df)} segments totaling up to {required_duration / 3600:.1f} hours")

    # Phonemize and clean phonemes
    src_df['phonemes'] = src_df['normalized_text'].progress_apply(phonemize_text)
    src_df['phonemes'] = src_df['phonemes'].progress_apply(remove_space_around_punctuation)
    print("Phonemization complete.")
    
    # Normalize and resample wav files here for the filtered segments
    for fid in tqdm(src_df.index):
        wav_in = src_wav_path / f"{fid}.wav"
        wav_out = dst_wav_path / f"{fid}.wav"
        audio = pydub.AudioSegment.from_file(wav_in)
        audio = audio.set_frame_rate(24000).set_channels(1).normalize()
        audio.export(wav_out, format="wav")

    # Prepare output
    src_df.reset_index(inplace=True)
    src_df['speaker_id'] = 0
    output_df = src_df[['file_id', 'phonemes', 'speaker_id', 'text']].sort_values(by='file_id')

    output_df.to_csv(dst_metadata_path, sep='|', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    main(Path(args.input), Path(args.output))
