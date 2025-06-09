import pandas as pd
from pathlib import Path

base_path = Path("./24khz_dataset")
metadata_csv = base_path / "metadata.csv"

# Load data without shuffling
df = pd.read_csv(metadata_csv, sep='|', header=None)

# Split sizes
val_size = int(len(df) * 0.1) # 10% for validation
train_size = len(df) - val_size

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# Save splits back to the same folder
train_df.to_csv(base_path / "train.txt", sep='|', header=False, index=False)
val_df.to_csv(base_path / "val.txt", sep='|', header=False, index=False)
