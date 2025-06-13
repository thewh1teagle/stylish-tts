import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Paths
base_path = Path('/workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced')
csv_path = base_path / 'train.txt'

# Read the file, assuming '|' separator and columns:
# file_id.wav | phonemes | sid | text
df = pd.read_csv(csv_path, sep='|', header=None, names=['file', 'phonemes', 'sid', 'text'])

# Strip spaces if any around columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Count phonemes with progress bar
phoneme_counter = Counter()

for idx, phoneme_str in enumerate(tqdm(df['phonemes'], desc="Counting phoneme characters")):
    # Debug: print phoneme string occasionally to check data correctness
    if idx % 10000 == 0:
        print(f"Line {idx} sample phonemes: {phoneme_str}")

    for ch in phoneme_str:
        phoneme_counter[ch] += 1
        # Debug specific characters, e.g. '?'
        if ch == '?':
            print(f"Found '?' in line {idx}: {phoneme_str}")

# Debug info summary
print(f"Total phoneme characters counted: {sum(phoneme_counter.values())}")
print(f"Unique phoneme characters: {len(phoneme_counter)}")

# Check counts for suspicious characters
print(f"Count of '?': {phoneme_counter.get('?', 0)}")
print(f"Count of 'x': {phoneme_counter.get('x', 0)}")
print(f"Count of 'a': {phoneme_counter.get('a', 0)}")  # Example phoneme
print(f"Count of ' ': {phoneme_counter.get(' ', 0)}")  # Count spaces if any

# Show top 10 most common phoneme characters
print("Top 10 most common phoneme characters:")
for ch, count in phoneme_counter.most_common(10):
    print(f"'{ch}': {count}")

# Convert counter to dict sorted by descending count
phoneme_counts = dict(sorted(phoneme_counter.items(), key=lambda item: item[1], reverse=True))

# Save JSON file
json_path = base_path / 'phoneme_distribution.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(phoneme_counts, f, ensure_ascii=False, indent=2)

# Plot distribution
plt.figure(figsize=(12, 6))
phonemes = list(phoneme_counts.keys())
counts = list(phoneme_counts.values())

plt.bar(phonemes, counts, color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Phonemes')
plt.ylabel('Count')
plt.title('Phoneme Distribution')
plt.tight_layout()

# Save plot
plot_path = base_path / 'phoneme_distribution.png'
plt.savefig(plot_path)
plt.close()

print(f"Phoneme distribution saved as JSON: {json_path}")
print(f"Phoneme distribution plot saved as PNG: {plot_path}")

print("Example lines containing '?':")
for phoneme_str in df['phonemes']:
    if '?' in phoneme_str:
        print(phoneme_str)
        break

print("Example lines containing 'x':")
for phoneme_str in df['phonemes']:
    if 'x' in phoneme_str:
        print(phoneme_str)
        break
