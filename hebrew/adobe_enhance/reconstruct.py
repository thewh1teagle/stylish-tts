"""
uv run reconstruct_segments.py enhanced segments merged
"""
import argparse
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("enhanced", type=Path, help="Folder with enhanced WAV files (e.g., 1.wav, 2.wav)")
parser.add_argument("metadata", type=Path, help="Folder with metadata JSON files (e.g., 1.json, 2.json)")
parser.add_argument("output", type=Path, help="Output folder for reconstructed WAVs")
args = parser.parse_args()

enhanced_path = args.enhanced
metadata_path = args.metadata
output_path = args.output
output_path.mkdir(exist_ok=True)


# Find all enhanced chunks (1.wav, 2.wav, etc.)
enhanced_files = sorted(enhanced_path.glob('*.wav'), key=lambda x: int(x.stem))

for enhanced_file in tqdm(enhanced_files, desc="Reconstructing"):
    index = enhanced_file.stem
    meta_file = metadata_path / f"{index}.json"
    
    if not meta_file.exists():
        print(f"⚠️ No metadata for {enhanced_file.name}, skipping.")
        continue

    with open(meta_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    audio = AudioSegment.from_wav(enhanced_file)

    for segment in segments:
        start_ms = int(segment['start_sec'] * 1000)
        end_ms = int(segment['end_sec'] * 1000)
        original_name = segment['filename']

        # Reconstruct and save
        clip = audio[start_ms:end_ms]
        clip.export(output_path / original_name, format='wav')

print("✅ Done. Reconstructed clips saved in ./reconstructed")