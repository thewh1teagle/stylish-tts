#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa
import gc
import json

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
import utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"()“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print("Meld " + char + ": " + text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300, sample_rate=24000)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 frame_count=None,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS
        self.cache = {}
        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path
        self.frame_count = frame_count

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id, mel_tensor = self._cache_tensor(data)
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # get OOD text
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)
        
        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)

        pad_start = 5000
        pad_end = 5000
        if self.frame_count is not None:
            pad_start = (self.frame_count*300 - wave.shape[0]) // 2
            pad_end = self.frame_count*300 - wave.shape[0] - pad_start
        wave = np.concatenate([np.zeros([pad_start]), wave, np.zeros([pad_end])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _cache_tensor(self, data):
        path = data[0]
        #if path in self.cache:
        #(wave, text_tensor, speaker_id, mel_tensor) = self.cache[path]
        #else:
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()
        #self.cache[path] = (wave, text_tensor, speaker_id,
        #                    mel_tensor)
        return (wave, text_tensor, speaker_id, mel_tensor)

    def _load_data(self, data):
        wave, text_tensor, speaker_id, mel_tensor = self._cache_tensor(data)

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     frame_count = None):
    
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, frame_count=frame_count, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=min(batch_size, len(dataset)),
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

class BatchManager:
    def __init__(self, train_bin_count, train_path, log_dir,
                 probe_batch=False, root_path="", OOD_data=[],
                 min_length=50, device="cpu",
                 accelerator=None, log_print=None):
        self.train_bin_count = train_bin_count
        self.train_path = train_path
        self.train_max = 0
        self.train_total_steps = 1
        self.probe_batch = probe_batch
        self.batch_dict = {}
        self.log_dir = log_dir

        self.root_path = root_path
        self.OOD_data = OOD_data
        self.min_length = min_length
        self.device = device
        self.accelerator = accelerator
        self.log_print = log_print

        if not self.probe_batch:
            self.init_for_training()

    def build_loader(self, train_list, batch_size, frame_count):
        loader = build_dataloader(train_list,
                                  self.root_path,
                                  OOD_data=self.OOD_data,
                                  min_length=self.min_length,
                                  batch_size=batch_size,
                                  num_workers=16,
                                  dataset_config={},
                                  device=self.device,
                                  frame_count=frame_count)
        if self.accelerator is not None:
            return self.accelerator.prepare(loader)
        else:
            return loader
    
    def init_for_training(self):
        batch_file = osp.join(self.log_dir, "batch_sizes.json")
        if osp.isfile(batch_file):
            with open(batch_file, "r") as batch_input:
                self.batch_dict = json.load(batch_input)
        
        for i in range(self.train_bin_count):
            train_list = utils.get_data_path_list(
                "%s/list-%d.txt" % (self.train_path, i))
            frame_count = get_frame_count(i)
            batch_size = self.get_batch_size(i)
            self.train_max += len(train_list)//batch_size
            self.train_total_steps += len(train_list)

    def get_batch_size(self, i):
        batch_size = 1
        if str(i) in self.batch_dict:
            batch_size = self.batch_dict[str(i)]
        return batch_size

    def set_batch_size(self, i, batch_size):
        self.batch_dict[str(i)] = batch_size

    def save_batch_dict(self):
        batch_file = osp.join(self.log_dir, "batch_sizes.json")
        with open(batch_file, "w") as o:
            json.dump(self.batch_dict, o)

    def epoch_loop(self, train_batch):
        if self.probe_batch:
            self.probe_loop(train_batch)
        else:
            self.train_loop(train_batch)

    def probe_loop(self, train_batch):
        self.batch_dict = {}
        batch_size = 100
        for i in range(self.train_bin_count):
            train_list = utils.get_data_path_list(
                "%s/list-%d.txt" % (self.train_path, i))
            if len(train_list) < 2:
                self.batch_dict[str(i)] = 0
                continue
            frame_count = get_frame_count(i)
            done = False
            while not done:
                try:
                    if batch_size > 0:
                        loader = self.build_loader(train_list,
                                                   batch_size,
                                                   frame_count)
                        print("Attempting %d/%d @ %d"
                              % (frame_count,
                                 get_frame_count(self.train_bin_count),
                                 batch_size))
                        for _, batch in enumerate(loader):
                            _, _ = train_batch(0, batch, 0, 0)
                            break
                    self.set_batch_size(i, batch_size)
                    
                    done = True
                except Exception as e:
                    print("Probe failed", e)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    counting_up = False
                    batch_size -= 1
        self.save_batch_dict()
        quit()

    def train_loop(self, train_batch):
        running_loss = 0
        iters = 0
        train_count = 0
        bin_list = random.sample(range(self.train_bin_count),
                                 self.train_bin_count)
        for i in bin_list:
            train_list = utils.get_data_path_list(
                "%s/list-%d.txt" % (self.train_path, i))
            if len(train_list) < 2:
                print("Skipping bin ", i, " (too short or doesn't exits)")
                continue
            batch_size = self.get_batch_size(i)
            if batch_size == 0:
                print("Skipping bin ", i, " (batch_size 0)")
            frame_count = get_frame_count(i)
            #print("Training bin %d (%d frames) @ batch_size %d"
            #      % (i, frame_count, batch_size))
            loader = self.build_loader(train_list,
                                       batch_size,
                                       frame_count)
            for _, batch in enumerate(loader):
                try:
                    running_loss, iters = train_batch(
                        train_count, batch, running_loss, iters)
                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        self.log_print("TRAIN_BATCH OOM ("
                                       + str(i) + ") @ batch_size "
                                       + str(batch_size))
                        batch_size -= 1
                        self.set_batch_size(i, batch_size)
                        self.save_batch_dict()
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                    else:
                        raise e
                train_count += 1

def get_frame_count(i):
    return i*4 + 20 + 34
