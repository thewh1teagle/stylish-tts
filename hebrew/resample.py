import sys
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

def main(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if output_path.exists():
        raise FileExistsError(f"Output folder '{output_path}' already exists. Please remove it or specify a different folder.")
    output_path.mkdir(parents=True, exist_ok=False)

    wav_files = list(input_path.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_path}")
        return

    for wav_file in tqdm(wav_files, desc="Resampling WAV files"):
        audio = AudioSegment.from_file(wav_file)
        audio = audio.set_frame_rate(24000).set_channels(1)  # resample + mono
        output_file = output_path / wav_file.name
        audio.export(output_file, format="wav")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_folder> <output_folder>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
