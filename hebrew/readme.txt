1. Setup environment
    pip install uv
    uv venv -p3.11
    uv pip install datasets soundfile librosa numpy tqdm

Note: you need at least 16GB GPU VRAM

2. Fetch dataset and phonemize
    uv run hebrew/fetch_dataset.py
    uv run hebrew/phonemize.py

Dataset fetched from https://huggingface.co/datasets/shb777/gemini-flash-2.0-speech

Note: dataset should be 24khz, mono, 16 bit.


3. Split to train.txt and val.txt with 10% for val
    uv run hebrew/split_dataset.py

4. Create pitch data, expect X minutes on RTX4090

git clone https://github.com/Stylish-TTS/stylish-dataset.git
cd stylish-dataset
uv sync
uv run stylish-dataset/all-pitch.py --wavdir ../24khz_dataset/wav --trainpath ../24khz_dataset/train.txt --valpath ../24khz_dataset/val.txt --split $(nproc) # use all cores

5. Setup the configs in config.yml under datasets. just change to your new absolute paths

6. Create alignment data
    mkdir checkpoints
    cd train
    uv sync
    uv run stylish_train/train.py \
        --model_config_path ../config/model.yml \
        --config_path ../config/config.yml \
        --stage alignment \
        --out_dir ../checkpoints

7. In config.yml 
    you should set epoch acoustic to 20 and textual epochs to 30 (related to the size of the dataset) 
    Also set probe batch max to 8 and 16, batch max is related to VRAM 


8. Create the actual alignment data using the trained model for alignment

cd train
uv pip install git+https://github.com/resemble-ai/monotonic_align.git@c6e5e6
PYTHONPATH=. uv run stylish_train/dataprep/align_text.py \
    --model_config_path ../config/model.yml \
    --config_path ../config/config.yml \
    --model ../checkpoints/alignment_model.safetensors \
    --out ../checkpoints/alignment.safetensors


 Upload pitch data to HuggingFace
