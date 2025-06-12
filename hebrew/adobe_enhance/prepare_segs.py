"""
uv venv
uv pip install pydub tqdm librosa
https://podcast.adobe.com/en/enhance
uv run prepare_segments.py input output
"""
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path, help="Input folder with .wav files")
parser.add_argument("output", type=Path, help="Output folder for merged files")
args = parser.parse_args()

src_path = args.input
dst_path = args.output
dst_path.mkdir(parents=True, exist_ok=True)

# Constants
MAX_CHUNK_DURATION = 60 * 60 * 1000  # 60 minutes in ms
SILENCE_DURATION = 2000  # 2 seconds in ms
silence = AudioSegment.silent(duration=SILENCE_DURATION)

# Load and sort files
wav_files = sorted(src_path.glob('*.wav'), key=lambda item: str(item))

chunk_index = 1
current_chunk = AudioSegment.silent(duration=0)
metadata = []
current_time = 0

for wav_file in tqdm(wav_files, desc="Processing WAV files"):
    audio = AudioSegment.from_wav(wav_file)
    duration_ms = len(audio)

    # If adding this file + 2 sec silence exceeds max, finalize current chunk
    if len(current_chunk) + duration_ms + SILENCE_DURATION > MAX_CHUNK_DURATION and metadata:
        out_wav = dst_path / f"{chunk_index}.wav"
        out_json = dst_path / f"{chunk_index}.json"
        current_chunk.export(out_wav, format="wav")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        chunk_index += 1
        current_chunk = AudioSegment.silent(duration=0)
        metadata = []
        current_time = 0

    # Append audio and metadata
    current_chunk += audio + silence
    metadata.append({
        "filename": wav_file.name,
        "start_sec": current_time / 1000,
        "end_sec": (current_time + duration_ms) / 1000
    })
    current_time += duration_ms + SILENCE_DURATION

# Save final chunk
if metadata:
    out_wav = dst_path / f"{chunk_index}.wav"
    out_json = dst_path / f"{chunk_index}.json"
    current_chunk.export(out_wav, format="wav")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ Done. Exported merged chunks and metadata.")