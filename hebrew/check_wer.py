"""
uv pip install pandas faster-whisper jiwer tqdm
install cudacnn 9.1.0 developer.nvidia.com/cudnn-9-1-0-download-archive
"""
import pandas as pd
import json
from faster_whisper import WhisperModel
from jiwer import wer, cer
from pathlib import Path
from tqdm import tqdm

# Paths
base_path = Path('/workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced')
csv_path = base_path / 'train.txt'
wav_path = base_path / 'wav'

# Load dataset
df = pd.read_csv(csv_path, sep='|', header=None, names=['filename', 'phonemes', 'sid', 'text'])

# Whisper model
model_size = "large-v3-turbo"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

results = []

# Process each file
for idx, row in tqdm(df.iterrows(), total=len(df)):
    audio_file = wav_path / f"{row['filename']}"
    if not audio_file.exists():
        print(f"Warning: Audio file not found: {audio_file}")
        continue

    try:
        segments, _ = model.transcribe(str(audio_file), beam_size=5)
        hyp_text = ' '.join([s.text for s in segments]).strip()

        ref_text = row['text'].strip()
        sample_wer = wer(ref_text, hyp_text)
        sample_cer = cer(ref_text, hyp_text)

        results.append({
            "filename": row['filename'],
            "ref": ref_text,
            "hyp": hyp_text,
            "wer": sample_wer,
            "cer": sample_cer
        })
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Sort results by filename
results.sort(key=lambda r: r["filename"])

# Calculate mean WER/CER
mean_wer = sum(r["wer"] for r in results) / len(results)
mean_cer = sum(r["cer"] for r in results) / len(results)

report = {
    "mean_wer": mean_wer,
    "mean_cer": mean_cer,
    "details": results
}

# Save to JSON
output_path = base_path / "wer_report.json"
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Saved WER report to {output_path}")
