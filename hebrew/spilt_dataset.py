import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()

base_path = Path(args.input)
metadata_csv = base_path / "metadata.csv"

# Load data without shuffling
df = pd.read_csv(metadata_csv, sep='|', header=None)

# Split sizes
val_size = int(len(df) * 0.1)  # 10% for validation
train_size = len(df) - val_size

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# Assert no overlap between train and val
assert train_df.index.intersection(val_df.index).empty, "Train and validation sets overlap!"

# Save splits back to the same folder
train_path = base_path / "train.txt"
val_path = base_path / "val.txt"

train_df.to_csv(train_path, sep='|', header=False, index=False)
val_df.to_csv(val_path, sep='|', header=False, index=False)

print(f"Train split saved to: {train_path}")
print(f"Validation split saved to: {val_path}")
