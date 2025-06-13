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
    uv run hebrew/spilt_dataset.py /workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced

4. Create pitch data, expect 50 minutes on RTX4090

git clone https://github.com/thewh1teagle/stylish-dataset
cd stylish-dataset
uv sync

# use all cores
uv run stylish-dataset/all-pitch.py --wavdir /workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced/wav \
    --trainpath /workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced/train.txt \
    --valpath /workspace/stylish-tts/ljspeech-enhanced/dataset_enhanced/val.txt \
    --split $(nproc) 

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
        --checkpoint ../checkpoints/acoustic/latest_checkpoint_dir
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


And when finetuning, you need to pass --reset_stage true 



# Fetch folders from huggingface
huggingface-cli download --repo-type model thewh1teagle/stylish-tts --include textual_checkpoint_00007_step_000039381/ --local-dir .

# upload model
uv run huggingface-cli upload --repo-type model thewh1teagle/stylish-tts ./ckpt/path # upload contents of the folder

























It failed with these lines in train.txt


LJ008-0050.wav|ɐ bɹˈɔːdʃiːt dˈeɪɾᵻd ˈeɪpɹəl twˈɛntifˈɔːɹθ, sˈɛvəntˌiːn ˈeɪɾisˈɛvən, dᵻskɹˈaɪbɪŋ ɐn ˌɛksɪkjˈuːʃən ɔnðə nˈuːliɪnvˈɛntᵻd skˈæfoʊld bᵻfˌɔːɹ ðə dˈɛɾɚz dˈɔːɹ,|0|A broadsheet dated April twenty-fourth, seventeen eighty-seven, describing an execution on the newly-invented scaffold before the debtors door,
LJ008-0074.wav|ðæt ʌv fˈiːbiː hˈæɹɪs, hˌuː ɪn sˈɛvəntˌiːn ˈeɪɾiˈeɪt wʌz bɑːɹbˈɛɹiəsli ˈɛksᵻkjˌuːɾᵻd ænd bˈɜːnt bᵻfˌɔːɹ nˈuːɡeɪt fɔːɹ kˈɔɪnɪŋ.|0|that of Phoebe Harris, who in seventeen eighty-eight was barbariously executed and burnt before Newgate for coining.
LJ027-0050.wav|həmˈɑːlədʒi ðˈʌs mˈiːnz aɪdˈɛntᵻɾi ʌv stɹˈʌktʃɚ wˌɪtʃ ɪz ðə ɹɪzˈʌlt ʌv aɪdˈɛntᵻɾi ʌv pˈɛɹəntɪdʒ. ɪɾ ɪz ðə stˈæmp ʌv hɚɹˈɛdᵻɾi.|0|Homology thus means identity of structure which is the result of identity of parentage. It is the stamp of heredity.
LJ027-0053.wav|ðə mˈoʊst stɹˈaɪkɪŋ fˈækt ʌv sˈɪmɪlɚ stɹˈʌktʃɚɹ ɐmˌʌŋ plˈænts ænd ɐmˌʌŋ ˈænɪməlz ɪz ðɪ ɛɡzˈɪstəns əvə kˈɑːmən dʒˈɛnɚɹəl plˈæn ɪn ˌɛni ɡɹˈuːp.|0|The most striking fact of similar structure among plants and among animals is the existence of a common general plan in any group.
LJ027-0153.wav|ɪn ˈɔːɹdɚ təbi sˌoʊ ˈædᵻd tə səksˈɛsɪv spˈiːsiːz, ˈɛvɹi ˌɪndᵻvˈɪdʒuːəl dˈɪɹ bᵻlˈɔŋɪŋ tə lˈeɪɾɚ spˈiːsiːz wʌz ɹᵻkwˈaɪɚd tə ɹᵻpˈiːt ɪn hɪz ˈoʊn lˈaɪftaɪm|0|in order to be so added to successive species, every individual deer belonging to later species was required to repeat in his own lifetime
LJ036-0069.wav|ɪnstˈɛd ʌv wˈeɪɾɪŋ ðˈɛɹ, ˈɑːswəld ɐpˈæɹəntli wɛnt æz fˈɑːɹ ɐwˈeɪ æz hiː kʊd ænd bˈɔːɹdᵻd ðə fˈɜːst ˈoʊk klˈɪf bˈʌs wˌɪtʃ kˈeɪm ɐlˈɔŋ|0|Instead of waiting there, Oswald apparently went as far away as he could and boarded the first Oak Cliff bus which came along
LJ028-0296.wav|ðˈaʊ kˈʌvɚɹɪst ðə fˈaʊlɪst dˈiːdz wɪððə fˈɛɹəst pˈɑːsᵻbəl nˈeɪm, wˌɛn ðˈaʊ sˈeɪɪst ðˌaɪ mˈeɪmɪŋ ɪz tə hˈɛlp ˌaʊɚ sˈiːdʒ fˈɔːɹwɚd.|0|thou coverest the foulest deeds with the fairest possible name, when thou sayest thy maiming is to help our siege forward.


LJ028-0296.wav
LJ036-0069

  File "/workspace/stylish-tts/train/stylish_train/dataprep/align_text.py", line 84, in main
    trains, scores = calculate_alignments(
                     ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/stylish-tts/train/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/stylish-tts/train/stylish_train/dataprep/align_text.py", line 132, in calculate_alignments
    alignment, scores = torch_align(
                        ^^^^^^^^^^^^
  File "/workspace/stylish-tts/train/stylish_train/dataprep/align_text.py", line 165, in torch_align
    assert alignment[i] == blank or alignment[i] == text[0, text_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError





Note: maybe for longer datasets alignment needed more than 20 epochs, more like 100 to converge.