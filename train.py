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
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


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

    train.logger = logging.getLogger(__name__)
    train.logger.setLevel(logging.DEBUG)
    handler = StreamHandler()
    handler.setLevel(logging.DEBUG)
    train.logger.addHandler(handler)

    train.log_dir = train.config["log_dir"]
    if not osp.exists(train.log_dir):
        os.makedirs(train.log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(train.log_dir, osp.basename(config_path)))
    # write logs
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

    train.val_interval = train.config.get("val_interval", 1)
    train.save_interval = train.config.get("save_interval", 1)

    train.data_params = train.config.get("data_params", None)
    train.sr = train.config["preprocess_params"].get("sr", 24000)
    train.train_path = train.data_params["train_data"]
    train.val_path = train.data_params["val_data"]
    train.root_path = train.data_params["root_path"]
    train.min_length = train.data_params["min_length"]
    train.OOD_data = train.data_params["OOD_data"]

    train.loss_params = Munch(train.config["loss_params"])
    train.diff_epoch = train.loss_params.diff_epoch
    train.joint_epoch = train.loss_params.joint_epoch
    train.TMA_epoch = train.loss_params.TMA_epoch

    train.precision = train.config.get("precision", "no")

    train.optimizer_params = Munch(train.config["optimizer_params"])

    if not osp.exists(train.train_path):
        print("Train data not found at {}".format(train.train_path))
        exit(1)
    if not osp.exists(train.val_path):
        print("Validation data not found at {}".format(train.val_path))
        exit(1)
    if not osp.exists(train.root_path):
        print("Root path not found at {}".format(train.root_path))
        exit(1)

    if "skip_downsamples" not in train.config["model_params"]:
        train.config["model_params"]["skip_downsamples"] = False
    train.model_params = recursive_munch(train.config["model_params"])
    train.multispeaker = train.model_params.multispeaker

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    train.accelerator = Accelerator(
        project_dir=train.log_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=train.precision,
    )

    if train.accelerator.is_main_process:
        train.writer = SummaryWriter(train.log_dir + "/tensorboard")

    train.device = train.config.get("device", "cuda")

    train.val_list = get_data_path_list(train.val_path)
    train.val_dataloader = build_dataloader(
        train.val_list,
        train.root_path,
        OOD_data=train.OOD_data,
        min_length=train.min_length,
        batch_size={},
        validation=True,
        num_workers=0,
        device=train.device,
        dataset_config={},
        multispeaker=train.multispeaker,
    )

    train.val_dataloader = train.accelerator.prepare(train.val_dataloader)

    def log_print_function(s):
        log_print(s, train.logger)

    train.batch_manager = BatchManager(
        train.train_path,
        train.log_dir,
        probe_batch=probe_batch,
        root_path=train.root_path,
        OOD_data=train.OOD_data,
        min_length=train.min_length,
        device=train.device,
        accelerator=train.accelerator,
        log_print=log_print_function,
        multispeaker=train.multispeaker,
    )

    with train.accelerator.main_process_first():
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

    for k in train.model:
        train.model[k] = train.accelerator.prepare(train.model[k])

    _ = [train.model[key].to(train.device) for key in train.model]

    # DP
    for key in train.model:
        if key != "mpd" and key != "msd" and key != "wd":
            train.model[key] = MyDataParallel(train.model[key])

    train.start_epoch = 1
    train.iters = 0

    load_pretrained = train.config.get("pretrained_model", "")

    # load an existing model for first stage
    if (
        load_pretrained
        and osp.exists(load_pretrained)
        and not train.config.get("second_stage_load_pretrained", False)
    ):
        print(f"Loading the first stage model at {load_pretrained} ...")
        train.model, _, train.start_epoch, train.iters = load_checkpoint(
            train.model,
            None,
            load_pretrained,
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
        train.start_epoch = 1
        train.model.predictor_encoder = copy.deepcopy(train.model.style_encoder)

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

    scheduler_params = {
        "max_lr": train.optimizer_params.lr,
        "pct_start": float(0),
        "epochs": train.epochs,
        "steps_per_epoch": train.batch_manager.get_step_count(),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in train.model}
    scheduler_params_dict["bert"]["max_lr"] = train.optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = train.optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = train.optimizer_params.ft_lr * 2

    train.optimizer = build_optimizer(
        {key: train.model[key].parameters() for key in train.model},
        scheduler_params_dict=scheduler_params_dict,
        lr=train.optimizer_params.lr,
    )

    for k in train.optimizer.optimizers.keys():
        train.optimizer.optimizers[k] = train.accelerator.prepare(
            train.optimizer.optimizers[k]
        )
        train.optimizer.schedulers[k] = train.accelerator.prepare(
            train.optimizer.schedulers[k]
        )

    # adjust BERT learning rate
    for g in train.optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = train.optimizer_params.bert_lr
        g["initial_lr"] = train.optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in train.optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = train.optimizer_params.ft_lr
            g["initial_lr"] = train.optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    if train.accelerator.main_process_first():
        # load models if there is a model for second stage
        if (
            load_pretrained
            and osp.exists(load_pretrained)
            and train.config.get("second_stage_load_pretrained", False)
        ):
            train.model, train.optimizer, train.start_epoch, train.iters = (
                load_checkpoint(
                    train.model,
                    train.optimizer,
                    load_pretrained,
                    load_only_params=train.config.get("load_only_params", True),
                )
            )
            train.start_epoch += 1

    train.n_down = train.model.text_aligner.n_down

    train.best_loss = float("inf")  # best test loss
    train.loss_train_record = list([])
    train.loss_test_record = list([])
    train.iters = 0

    train.criterion = nn.L1Loss()  # F0 loss (regression)
    torch.cuda.empty_cache()

    train.stft_loss = MultiResolutionSTFTLoss().to(train.device)

    # print("BERT", optimizer.optimizers["bert"])
    # print("decoder", optimizer.optimizers["decoder"])

    train.start_ds = False

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

    train_val_loop(train)


def train_val_loop(train):
    if train.stage == "first":
        train.train_batch = train_first
        train.validate = validate_first
    elif train.stage == "second":
        train.train_batch = train_second
        train.validate = validate_second
    else:
        exit("Invalid training stage. --stage must be 'first' or 'second'")
    for epoch in range(train.start_epoch, train.epochs):
        train.running_loss = 0
        train.start_time = time.time()

        if epoch >= train.diff_epoch or train.early_joint:
            train.start_ds = True

        _ = [train.model[key].train() for key in train.model]
        train.batch_manager.epoch_loop(epoch, train=train)
        _ = [train.model[key].eval() for key in train.model]
        train.validate(epoch, 1, True, train)


if __name__ == "__main__":
    main()
