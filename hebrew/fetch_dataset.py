"""
uv run hebrew/fetch_dataset.py

Get duration of the folder using:

    soxi -DT ./24khz_dataset/wav/*.wav
"""
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import pandas as pd
import sys

data_path = Path('24khz_dataset')
wav_path = data_path / 'wav'
metadata_path = data_path / 'metadata_text.csv'
wav_path.mkdir(exist_ok=True, parents=True)

required_dur = 3600 * 8  # 8 hours
max_sample_dur = 10  # skip samples longer than this

dataset = load_dataset('shb777/gemini-flash-2.0-speech', streaming=True)
dataset_iter = iter(dataset['en'])

fetched_dur = 0
df = pd.DataFrame(columns=[0, 1])

i = 0
for row in dataset_iter:
    puck_data = row['puck']
    samples = puck_data['array']
    sample_rate = puck_data['sampling_rate']
    dur = len(samples) / sample_rate

    if dur > max_sample_dur:
        print(f"⏭️ Skipping {i} (too long: {dur:.2f}s 🎵)")
        continue

    audio_path = wav_path / f'{i}.wav'
    sf.write(audio_path, samples, sample_rate)

    df.loc[len(df)] = [i, row['text']]
    fetched_dur += dur

    percent = int((fetched_dur / required_dur) * 100)
    print(f"📈 {percent}% done — total: {fetched_dur:.1f}s / {required_dur}s")

    print(f"✅ Added {i} ({dur:.2f}s 🎵)")
    df.to_csv(metadata_path, sep='|', index=False, header=False)

    i += 1

    if fetched_dur >= required_dur:
        print(f"🎯 Target reached: {fetched_dur:.2f}s collected! Stopping. 🛑")
        break

print("🎉 Done fetching all samples! 🎉")
sys.exit(0) # Otherwise the GIL locked for some reason
