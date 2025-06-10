import pandas as pd

def filter_low_confidence(
    score_path: str,
    transcript_path: str,
    output_path: str,
    quantile: float = 0.10
):
    """
    Removes the bottom X% of samples based on confidence scores.

    Args:
        score_path (str): Path to scores file (e.g. scores_train.txt).
        transcript_path (str): Path to transcript file (e.g. train.txt).
        output_path (str): Path to save filtered output (e.g. train_filtered.txt).
        quantile (float): Quantile cutoff to remove (default 0.10 for bottom 10%).
    """
    print(f"📊 Loading scores from: {score_path}")
    scores_df = pd.read_csv(score_path, sep=" ", header=None, names=["score", "filename"])
    
    threshold = scores_df["score"].quantile(quantile)
    print(f"🔻 Removing entries with score <= {threshold:.4f}")
    
    filtered_filenames = scores_df[scores_df["score"] > threshold]["filename"].tolist()
    print(f"✅ Keeping {len(filtered_filenames)} of {len(scores_df)} entries")

    print(f"📄 Loading transcripts from: {transcript_path}")
    transcript_df = pd.read_csv(transcript_path, sep="|", header=None,
                                names=["filename", "phonemes", "speaker", "text"])
    
    filtered_df = transcript_df[transcript_df["filename"].isin(filtered_filenames)]

    print(f"💾 Saving filtered data to: {output_path} ({len(filtered_df)} entries)")
    filtered_df.to_csv(output_path, sep="|", index=False, header=False)

# === Apply to train and val ===

filter_low_confidence(
    score_path="/workspace/stylish-tts/checkpoints/scores_train.txt",
    transcript_path="/workspace/stylish-tts/24khz_dataset/train.txt",
    output_path="/workspace/stylish-tts/24khz_dataset/train_filtered.txt"
)

filter_low_confidence(
    score_path="/workspace/stylish-tts/checkpoints/scores_val.txt",
    transcript_path="/workspace/stylish-tts/24khz_dataset/val.txt",
    output_path="/workspace/stylish-tts/24khz_dataset/val_filtered.txt"
)
