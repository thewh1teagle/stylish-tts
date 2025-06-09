1. Setup environment
    pip install uv
    uv venv -p3.11
    uv pip install datasets soundfile librosa numpy tqdm

Note: you need at least 16GB GPU VRAM

2. Fetch dataset and phonemize
    uv run hebrew/fetch_dataset.py
    uv run hebrew/phonemize.py

Note: dataset should be 24khz, mono, 16 bit.


3. Split to train.txt and val.txt with 10% for val
    uv run hebrew/split_dataset.py

4. Create pitch data, expect X minutes on RTX4090

git clone https://github.com/Stylish-TTS/stylish-dataset.git
cd stylish-dataset
uv sync
uv run stylish-dataset/all-pitch.py --wavdir ../24khz_dataset/wav --trainpath ../24khz_dataset/train.txt --valpath ../24khz_dataset/val.txt --split $(nproc) # use all cores

5. Upload pitch data to HuggingFace
