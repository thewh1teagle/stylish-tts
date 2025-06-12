git clone https://github.com/thewh1teagle/stylish-tts -b hebrew-v1
cd stylish-tts
sudo apt-get install aria2 -y
aria2c -x16 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xf LJSpeech-1.1.tar.bz2

pip install uv
uv venv -p3.12
uv pip install soundfile librosa numpy tqdm pydub phonemizer-fork espeakng-loader pandas

uv run hebrew/ljspeech/prepare_dataset.py --input ./LJSpeech-1.1 --output ./dataset



uv run hebrew/adobe_enhance/prepare_segs.py ./dataset/wav ./dataset_before_enhance
uv run hebrew/adobe_enhance/reconstruct.py ./dataset_after_enhance ./dataset_before_enhance ./dataset_enhanced_segments
cp ./dataset/metadata.csv ./dataset_enhanced_segments/metadata.csv

Finally do uv export 



sudo apt install -y sox
soxi -DT *.wav