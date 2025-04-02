# Stylish TTS

Stylish TTS is a lightweight text-to-speech system suitable for offline local use. Our goal is providing consistency for long form text and screen reading with a focus on high quality single speaker models rather than zero-shot voice cloning. The architecture is based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) with many bugfixes, model improvements, and a streamlined training process.

TODO: Make some samples

# Getting Started

## Dependencies

The biggest dependency is of course the nvidia/CUDA drivers/system.

After those are installed, you can run the training code using uv which will take care of all the other python dependencies. Install uv via:

```
pipx install uv
pipx ensure-path
```

# Training a Model

In order to train your model, you need a GPU with at least 16 GB of VRAM and PyTorch support and you will need a dataset.

## Preparing a dataset

A dataset consists of many segments. Each segment has a written text and an audio file where that text is spoken by a reader. When using the default model options, the audio must be at least 0.25 seconds long. The upper limit on audio length for a segment will be based on the VRAM of your GPU. You typically want to have audio clips distributed over the whole range of possible lengths. If your range doesn't cover the shortest lengths, your model will sound worse when doing short utterances of one word or a few words. If your range doesn't cover longer lengths which include multiple sentences, your model will tend to skip past punctuation too quickly.

### Segment Distribution

Segments must have 510 phonemes or less. Audio segments must be at least 0.25 seconds long. The upper limit on audio length is determined by your VRAM and the training stage. Generally speaking, you will want to have a distribution of segments between 0.25 seconds and 10 seconds long. If you have a the VRAM, you can include even longer segments, though there are diminishing returns.

### Training List / Validation List

Training and validation lists are a series of lines in the following format:

`<filename>|<phonemes>|<speaker-id>|<plaintext>`

The filename for the segment audio and should be a .wav file (24 khz, mono) in the wav_path from your config.yml.

The phonemes are the IPA representation of how your segment text is pronounced.

Speaker ID is an arbitrary integer which should be applied to every segment that has the same speaker. For single-speaker datasets, this will typically always be '0'.

The plaintext is the original text of your utterance before phonemization. It does not need to be tokenized or normalized, but obviously should not include the '|' character.

### Pitch Data

Stylish TTS uses a pre-cached ground truth pitch (F0) for all your segments. There is a script to generate it available at the stylish-datasets repository:

https://github.com/Stylish-TTS/stylish-dataset

calculate-pitch.py is a single-process version while all-pitch.py calculates them in parallel using multi-processing. Pitch is calculated using Harvest which is CPU-only and so it will take some time.

## Running train.py

Here is a typical command to start off a new training run using a single machine.

```
uv run stylish-tts/train.py \
    --model_config_path config/model.yml \
    --config_path /path/to/your/config.yml \
    --stage alignment \
    --out_dir /path/to/your/output
```

model_config_path: You should usually leave the model_config_path pointing at the default model configuration.

config_path: You should make your own copy of the config.yaml in the repository and fill in the paths to your dataset.

stage: It is best to start with alignment pre-training. It is very fast and makes all other stages train better. Later if you are loading a checkpoint, you may want to pick another stage to resume from.

out_dir: This is the destination path for all checkpoints, training logs, and tensorboard data. A separate sub-directory is created for each stage of training. Make sure to have plenty of disk space available here as checkpoints can take a large amount of storage.

It will take a long time to run this script. So it is a good idea to run using screen or tmux to have a persistent shell that won't disappear if you get disconnected or close the window.

Stages advance automatically and a checkpoint is created at the end of every stage before moving to the next. Other checkpoints will be saved and validations will be periodically run based on your config.yml settings.

## Loading a checkpoint

```
uv run stylish-tts/train.py \
    --model_config_path config/model.yml \
    --config_path /path/to/your/config.yml \
    --stage <stage>
    --out_dir /path/to/your/output \
    --checkpoint /path/to/your/checkpoint
```

You can load a checkpoint from any stage via the --checkpoint argument. You still need to set --stage appropriately to one of "alignment|pre_acoustic|acoustic|pre_textual|textual|joint". If you set it to the same stage as the checkpoint loaded from, it will continue in that stage at the same step number and epoch. If it is a different stage, it will train the entire stage.

Note that Stylish TTS checkpoints are not compatible with StyleTTS 2 checkpoints.

# Training New Languages

## Phonemization



