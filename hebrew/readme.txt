1. Setup environment
    pip install uv
    uv venv -p3.11
    uv pip install datasets soundfile librosa numpy tqdm

Note: you need at least 16GB GPU VRAM
Note: I trained on RTX4060 with Cuda version 12.1 and Python 3.11 on Ubuntu 22.04

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

Note: create issue to remove the chars [] from the dataset otherwise it failed with tensors issue. remove in dataset creation from the text itself/phonemes
Note: create issue that some line gave me tensor errors targets length is too long for CTC. found log_probs length: 100, targets length: 166 and num of repeats 3

9. Cut bad segments from the (Optional)
    uv run hebrew/remove_bad_segments.py

Then rename the files in the dataset folder...


10. Train the first stage (acustic)
    cd train
    uv run stylish_train/train.py \
        --model_config_path ../config/model.yml \
        --config_path ../config/config.yml \
        --stage acoustic \
        --out_dir ../checkpoints

Note: removed the line from train.txt in 3971.wav:
3971.wav|"""jɐ ɡɑːt klˈæs,ɹˈiːəl klˈæs,"" pɚfˈɔːɹməns tˈɛkst kˈoʊɹˈɪʔn̩ wɪð vˈɪki stˈɑːlsən,pɹədˈuːst æt lˌɑːs ˈændʒəlɪs kəntˈɛmpɚɹˌɛɹi ɛksɪbˈɪʃənz,nˈaɪntiːnhˈʌndɹɪd ˈeɪɾi."|0|"""Ya Got Class,Real Class,"" performance text co-written with Vicki Stolsen,produced at Los Angeles Contemporary Exhibitions,1980."

11. Train the second stage (textual)
    uv run stylish_train/train.py \
        --model_config_path ../config/model.yml \
        --config_path ../config/config.yml \
        --stage textual \
        --out_dir ../checkpoints \
        --checkpoint ../checkpoints/acustic/latest_checkpoint_dir
    cd train

11. Export onnx


cd train
uv run stylish_train/train.py \
    --convert true \
    --model_config_path ../config/model.yml \
    --config_path ../config/config.yml \
    --stage textual \
    --out_dir ./onnx_output \
    --checkpoint ./checkpoints/checkpoint_00002_step_000004919

In case you have tensors issue, print the file names and line in align_text.py then remove bad lines



Backup dataset, pitch data, and alignment data

sudo apt install p7zip-full -y
uv pip install huggingface_hub
git config --global credential.helper store # Allow clone private repo from HF
# Get token from https://huggingface.co/settings/tokens
uv run huggingface-cli login --token "token" --add-to-git-credential #

7z a 24khz_dataset.7z 24khz_dataset
uv run huggingface-cli upload --repo-type model stylish-tts ./24khz_dataset.7z
uv run huggingface-cli upload --repo-type model stylish-tts ./stylish-dataset/pitch.safetensors
uv run huggingface-cli upload --repo-type model stylish-tts ./checkpoints/alignment_model.safetensors
uv run huggingface-cli upload --repo-type model stylish-tts ./checkpoints/alignment.safetensors
uv run huggingface-cli upload --repo-type model stylish-tts ./checkpoints/acoustic/acoustic_batch_sizes.json 
uv run huggingface-cli upload --repo-type model stylish-tts ./checkpoints/alignment/alignment_batch_sizes.json 



For fine tune:
Do the same steps above (pitch,alignment model, alignment, align text) and then train normally but provide the checkpoint.