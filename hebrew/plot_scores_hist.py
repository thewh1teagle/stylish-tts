import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import librosa

# Load lines from the file
lines = Path("/workspace/stylish-tts/checkpoints/scores_train.txt").read_text().splitlines()

# Extract scores and audio file paths
scores = [float(line.split(' ')[0]) for line in lines]
audio_paths = ["/workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced/wav/" + line.split(' ', 1)[1] for line in lines]

# Compute audio lengths
audio_len = [librosa.get_duration(path=path) for path in audio_paths]

"""# Plot: Audio Length vs Score
plt.scatter(audio_len, scores, alpha=0.6, edgecolor='black')
plt.xlabel('Audio Length (seconds)')
plt.ylabel('Score')
plt.title('Audio Length vs. Score')
plt.grid(True)
plt.show()"""

# Define bins with width 0.1
bin_width = 0.05
data = scores
bins = np.arange(min(data), max(data) + bin_width, bin_width)

# Plot histogram
plt.hist(data, bins=bins, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Histogram with {bin_width} Bin Width')
plt.grid(True)
plt.savefig("score_histogram.png", dpi=300, bbox_inches='tight')
