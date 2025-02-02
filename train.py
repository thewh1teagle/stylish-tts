# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
import logging
from logging import StreamHandler

warnings.simplefilter("ignore")
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader, BatchManager

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer
from stages import train_first, validate_first, train_second, validate_second


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainContext:
    def __init__(self):
        pass


train = TrainContext()


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
@click.option("--probe_batch", default=None, type=int)
@click.option("--early_joint/--no_early_joint", default=False, type=bool)
@click.option("--stage", default="auto", type=str)
def main(config_path, probe_batch, early_joint, stage):
    train.config_path = config_path
    train.config = yaml.safe_load(open(config_path))
    train.early_joint = early_joint
    train.stage = stage

    train.log_dir = train.config["log_dir"]
    if not osp.exists(train.log_dir):
        os.makedirs(train.log_dir, exist_ok=True)
    if not osp.exists(train.log_dir):
        exit("Failed to create or find log directory.")
    shutil.copy(config_path, osp.join(train.log_dir, osp.basename(config_path)))
    train.writer = SummaryWriter(train.log_dir + "/tensorboard")

    train.logger = logging.getLogger(__name__)
    train.logger.setLevel(logging.DEBUG)
    handler = StreamHandler()
    handler.setLevel(logging.DEBUG)
    train.logger.addHandler(handler)
    file_handler = logging.FileHandler(osp.join(train.log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    )
    train.logger.addHandler(file_handler)

    train.epochs = train.config.get("epochs_2nd", 200)
    train.save_freq = train.config.get("save_freq", 2)
    train.log_interval = train.config.get("log_interval", 10)
    train.saving_epoch = train.config.get("save_freq", 2)

    train.sr = train.config["preprocess_params"].get("sr", 24000)

    train.loss_params = Munch(train.config["loss_params"])
    train.diff_epoch = train.loss_params.diff_epoch
    train.joint_epoch = train.loss_params.joint_epoch
    train.TMA_epoch = train.loss_params.TMA_epoch

    if "skip_downsamples" not in train.config["model_params"]:
        train.config["model_params"]["skip_downsamples"] = False
    train.model_params = recursive_munch(train.config["model_params"])
    train.multispeaker = train.model_params.multispeaker
    train.device = "cuda"

    # Set up data loaders and batch manager
    data_params = train.config.get("data_params", None)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    min_length = data_params["min_length"]
    OOD_data = data_params["OOD_data"]

    if not osp.exists(train_path):
        exit(f"Train data not found at {train_path}")
    if not osp.exists(val_path):
        exit(f"Validation data not found at {val_path}")
    if not osp.exists(root_path):
        exit("Root path not found at {root_path}")

    val_list = get_data_path_list(val_path)
    train.val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size={},
        validation=True,
        num_workers=4,
        device=train.device,
        dataset_config={},
        multispeaker=train.multispeaker,
    )

    def log_print_function(s):
        train.logger.info(s)

    train.batch_manager = BatchManager(
        train_path,
        train.log_dir,
        probe_batch=probe_batch,
        root_path=root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        device=train.device,
        accelerator=None,
        log_print=log_print_function,
        multispeaker=train.multispeaker,
    )

    # load pretrained ASR model
    ASR_config = train.config.get("ASR_config", False)
    ASR_path = train.config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = train.config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = train.config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    # build model
    train.model = build_model(train.model_params, text_aligner, pitch_extractor, plbert)
    _ = [train.model[key].to(train.device) for key in train.model]

    # DP
    for key in train.model:
        if key != "mpd" and key != "msd" and key != "wd":
            train.model[key] = MyDataParallel(train.model[key])

    start_epoch = 1
    train.iters = 0

    load_pretrained = train.config.get(
        "pretrained_model", ""
    ) != "" and train.config.get("second_stage_load_pretrained", False)

    if not load_pretrained:
        if train.config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(
                train.log_dir, train.config.get("first_stage_path", "first_stage.pth")
            )
            print("Loading the first stage model at %s ..." % first_stage_path)
            train.model, _, start_epoch, train.iters = load_checkpoint(
                train.model,
                None,
                first_stage_path,
                load_only_params=True,
                ignore_modules=[
                    "bert",
                    "bert_encoder",
                    "predictor",
                    "predictor_encoder",
                    "msd",
                    "mpd",
                    "wd",
                    "diffusion",
                ],
            )  # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            # diff_epoch += start_epoch
            # joint_epoch += start_epoch
            # epochs += start_epoch
            start_epoch = 1
            train.model.predictor_encoder = copy.deepcopy(train.model.style_encoder)
        else:
            start_epoch = 1
            train.iters = 0
            # raise ValueError("You need to specify the path to the first stage model.")

    train.gl = GeneratorLoss(train.model.mpd, train.model.msd).to(train.device)
    train.dl = DiscriminatorLoss(train.model.mpd, train.model.msd).to(train.device)
    train.wl = WavLMLoss(
        train.model_params.slm.model,
        train.model.wd,
        train.sr,
        train.model_params.slm.sr,
    ).to(train.device)

    train.gl = MyDataParallel(train.gl)
    train.dl = MyDataParallel(train.dl)
    train.wl = MyDataParallel(train.wl)

    train.sampler = DiffusionSampler(
        train.model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )

    optimizer_params = Munch(train.config["optimizer_params"])
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": train.epochs,
        "steps_per_epoch": train.batch_manager.get_step_count(),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in train.model}
    scheduler_params_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2

    train.optimizer = build_optimizer(
        {key: train.model[key].parameters() for key in train.model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    # adjust BERT learning rate
    for g in train.optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = optimizer_params.bert_lr
        g["initial_lr"] = optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in train.optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = optimizer_params.ft_lr
            g["initial_lr"] = optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    # load models if there is a model
    if load_pretrained:
        train.model, train.optimizer, start_epoch, train.iters = load_checkpoint(
            train.model,
            train.optimizer,
            train.config["pretrained_model"],
            load_only_params=train.config.get("load_only_params", True),
        )
        start_epoch += 1

    train.n_down = train.model.text_aligner.n_down

    train.best_loss = float("inf")  # best test loss

    torch.cuda.empty_cache()

    train.stft_loss = MultiResolutionSTFTLoss().to(train.device)

    # print("BERT", optimizer.optimizers["bert"])
    # print("decoder", optimizer.optimizers["decoder"])

    train.start_ds = False

    # TODO: This value is calculated inconsistently based on whether checkpoints are loaded/saved
    train.running_std = []

    train.slmadv_params = Munch(train.config["slmadv_params"])
    train.slmadv = SLMAdversarialLoss(
        train.model,
        train.wl,
        train.sampler,
        train.slmadv_params.min_len,
        train.slmadv_params.max_len,
        batch_percentage=train.slmadv_params.batch_percentage,
        skip_update=train.slmadv_params.iter,
        sig=train.slmadv_params.sig,
    )

    train_val_loop(train, start_epoch)


def train_val_loop(train, start_epoch):
    if train.stage == "first":
        train_batch = train_first
        validate = validate_first
    elif train.stage == "second":
        train_batch = train_second
        validate = validate_second
    else:
        exit("Invalid training stage. --stage must be 'first' or 'second'")
    for epoch in range(start_epoch, train.epochs):
        train.running_loss = 0
        train.start_time = time.time()

        if epoch >= train.diff_epoch or train.early_joint:
            train.start_ds = True

        _ = [train.model[key].train() for key in train.model]
        train.batch_manager.epoch_loop(epoch, train_batch, train=train)
        _ = [train.model[key].eval() for key in train.model]
        validate(epoch, 1, True, train)


if __name__ == "__main__":
    main()
