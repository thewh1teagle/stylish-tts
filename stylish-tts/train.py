# load packages
import time
import torch
import click
import shutil
import logging
import random
from logging import StreamHandler
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from config_loader import load_config_yaml, load_model_config_yaml
from train_context import TrainContext
from text_utils import TextCleaner

import numpy as np

#  warnings.simplefilter("ignore")
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader, FilePathDataset
from batch_manager import BatchManager

from models.models import build_model, load_defaults
from losses import GeneratorLoss, DiscriminatorLoss, WavLMLoss, MultiResolutionSTFTLoss
from utils import get_data_path_list


from models.slmadv import SLMAdversarialLoss
from models.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
)

import os.path as osp
import os

from optimizers import build_optimizer
from stages import (
    train_first,
    validate_first,
    train_second,
    validate_second,
    train_acoustic_adapter,
    train_vocoder_adapter,
)

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, etc.)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Customize the log message format
)

logger = logging.getLogger(__name__)  # Create a logger for the current module


def setup_logger(logger, out_dir):
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent messages from being passed to the root logger

    # Always add a stream handler
    err_handler = StreamHandler()
    err_handler.setLevel(logging.DEBUG)
    err_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(err_handler)

    # Always add a file handler
    file_handler = logging.FileHandler(osp.join(out_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


# simple fix for dataparallel that allows access to class attributes
# class MyDataParallel(torch.nn.DataParallel):
#    def __getattr__(self, name):
#        try:
#            return super().__getattr__(name)
#        except AttributeError:
#            return getattr(self.module, name)


@click.command()
@click.option("-p", "--config_path", default="configs/new.config.yml", type=str)
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--out_dir", type=str)
@click.option("--early_joint/--no_early_joint", default=False, type=bool)
@click.option(
    "--stage", default="first_tma", type=str
)  # "first", "first_tma", "second", "second_style", "second_joint"
@click.option("--checkpoint", default="", type=str)
def main(config_path, model_config_path, out_dir, early_joint, stage, checkpoint):
    train = TrainContext()
    np.random.seed(1)
    random.seed(1)
    if osp.exists(config_path):
        train_config = load_config_yaml(config_path)
        train.config = train_config
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)

    if osp.exists(model_config_path):
        train.model_config = load_model_config_yaml(model_config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {model_config_path}")
        exit(1)

    train.early_joint = early_joint
    train.base_output_dir = out_dir
    train.out_dir = osp.join(out_dir, stage)

    if not osp.exists(train.out_dir):
        os.makedirs(train.out_dir, exist_ok=True)

    if not osp.exists(train.out_dir):
        exit(f"Failed to create or find log directory at {train.out_dir}.")
    shutil.copy(config_path, osp.join(train.out_dir, osp.basename(config_path)))
    shutil.copy(
        model_config_path, osp.join(train.out_dir, osp.basename(model_config_path))
    )

    train.logger = logging.getLogger(__name__)
    setup_logger(train.logger, train.out_dir)

    train.manifest.max_epoch = sum(
        [
            train.config.training_plan.first,
            train.config.training_plan.first_tma,
            train.config.training_plan.second,
            train.config.training_plan.second_style,
            train.config.training_plan.second_joint,
        ]
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False, find_unused_parameters=True
    )
    train.accelerator = Accelerator(
        project_dir=train.out_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=train.config.training.mixed_precision,
    )
    train.accelerator.even_batches = False

    if train.accelerator.is_main_process:
        train.writer = SummaryWriter(train.out_dir + "/tensorboard")

    train.accelerator.register_for_checkpointing(train.config)
    train.accelerator.register_for_checkpointing(train.model_config)
    train.accelerator.register_for_checkpointing(train.manifest)

    # Set up data loaders and batch manager
    if not osp.exists(train.config.dataset.train_data):
        exit(f"Train data not found at {train.config.dataset.train_data}")
    if not osp.exists(train.config.dataset.val_data):
        exit(f"Validation data not found at {train.config.dataset.val_data}")
    if not osp.exists(train.config.dataset.wav_path):
        exit(f"Root path not found at {train.config.dataset.wav_path}")

    text_cleaner = TextCleaner(train.model_config.symbol)
    val_list = get_data_path_list(train.config.dataset.val_data)
    val_dataset = FilePathDataset(
        val_list,
        train.config.dataset.wav_path,
        OOD_data=train.config.dataset.OOD_data,
        min_length=train.config.dataset.min_length,
        validation=True,
        multispeaker=train.model_config.model.multispeaker,
        text_cleaner=text_cleaner,
    )
    train.val_dataloader = build_dataloader(
        val_dataset,
        val_dataset.time_bins(),
        validation=True,
        batch_size={},
        num_workers=4,
        device=train.config.training.device,
        multispeaker=train.model_config.model.multispeaker,
    )

    train.val_dataloader = train.accelerator.prepare(train.val_dataloader)

    train.batch_manager = BatchManager(
        train.config.dataset.train_data,
        train.out_dir,
        probe_batch_max=train.config.training.probe_batch_max,
        root_path=train.config.dataset.wav_path,
        OOD_data=train.config.dataset.OOD_data,
        min_length=train.config.dataset.min_length,
        device=train.config.training.device,
        accelerator=train.accelerator,
        multispeaker=train.model_config.model.multispeaker,
        text_cleaner=text_cleaner,
        stage=stage,
        epoch=train.manifest.current_epoch,
    )

    # build model
    train.model, kdiffusion = build_model(train.model_config)
    for key in train.model:
        train.model[key] = train.accelerator.prepare(train.model[key])
        train.model[key].to(train.config.training.device)

    # DP
    # for key in train.model:
    #    if key != "mpd" and key != "msd" and key != "wd":
    #        train.model[key] = MyDataParallel(train.model[key])

    # TODO: I think we want to start with epoch 0 or fix the tensorboard logging because it only writes gt sample when epoch 0
    train.manifest.current_epoch = 1
    train.manifest.current_total_step = 0

    scheduler_params = {
        "max_lr": train.config.optimizer.lr,
        "pct_start": float(0),
        # TODO: This should actually be based on the current stage
        "epochs": train.manifest.max_epoch,
        "steps_per_epoch": train.batch_manager.get_step_count(),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in train.model}
    # scheduler_params_dict["bert"]["max_lr"] = train.config.optimizer.bert_lr * 2
    # scheduler_params_dict["decoder"]["max_lr"] = train.config.optimizer.ft_lr * 2
    # scheduler_params_dict["style_encoder"]["max_lr"] = train.config.optimizer.ft_lr * 2

    train.optimizer = build_optimizer(
        {key: train.model[key].parameters() for key in train.model},
        scheduler_params_dict=scheduler_params_dict,
        lr=train.config.optimizer.lr,
    )

    if checkpoint:
        train.accelerator.load_state(checkpoint)
        train.config = train_config
        # if we are not loading on a epoch boundary we need to resume the loader and skip to the correct step
        if train.manifest.current_step != 0 and train.manifest.stage == stage:
            train.batch_manager.resume_loader = train.accelerator.skip_first_batches(
                train.batch_manager.loader, train.manifest.current_step
            )
        logger.info(f"Loading last checkpoint at {checkpoint} ...")
    else:
        # TODO: do we need this? We used to do it always before?
        load_defaults(train, train.model)

    # if we are changing stages we need to bump the epoch and reset the step
    if train.manifest.stage != stage:
        train.manifest.current_epoch += 1
        train.manifest.current_step = 0

    train.manifest.stage = stage

    # # load an existing model for first stage
    # if (
    #     pretrained_model
    #     and osp.exists(pretrained_model)
    #     and stage in ["first", "first_tma", "acoustic", "vocoder"]
    # ):
    #     logger.info(f"Loading the first stage model at {pretrained_model} ...")
    #     (
    #         train.model,
    #         train.optimizer,
    #         train.manifest.current_epoch,
    #         train.manifest.current_total_step,
    #     ) = load_checkpoint(
    #         train.model,
    #         train.optimizer,
    #         pretrained_model,
    #         # load_only_params=True,
    #         ignore_modules=[
    #             "bert",
    #             "bert_encoder",
    #             "predictor",
    #             "predictor_encoder",
    #             "msd",
    #             "mpd",
    #             "wd",
    #             "diffusion",
    #         ],
    #     )  # keep starting epoch for tensorboard log

    #     # TODO: what epoch are we on?
    #     # these epochs should be counted from the start epoch
    #     # diff_epoch += start_epoch
    #     # joint_epoch += start_epoch
    #     # epochs += start_epoch
    #     train.manifest.current_epoch = 1
    #     # TODO: This should happen only once when starting stage 2
    #     # train.model.predictor_encoder = copy.deepcopy(train.model.style_encoder)
    # elif stage in ["first", "first_tma", "acoustic", "vocoder"]:
    #     load_defaults(train, train.model)

    # # load models if there is a model for second stage
    # if (
    #     pretrained_model
    #     and osp.exists(pretrained_model)
    #     and stage in {"second", "second_style", "second_joint"}
    # ):
    #     (
    #         train.model,
    #         train.optimizer,
    #         train.manifest.current_epoch,
    #         train.manifest.current_total_step,
    #     ) = load_checkpoint(
    #         train.model, train.optimizer, pretrained_model, ignore_modules=[]
    #     )
    #     train.manifest.current_epoch += 1
    # elif stage in ["second", "second_style", "second_joint"]:
    #     load_defaults(train, train.model)

    train.generator_loss = GeneratorLoss(train.model.mpd, train.model.msd).to(
        train.config.training.device
    )
    train.discriminator_loss = DiscriminatorLoss(train.model.mpd, train.model.msd).to(
        train.config.training.device
    )
    train.wavlm_loss = WavLMLoss(
        train.model_config.slm.model,
        train.model.wd,
        train.model_config.preprocess.sample_rate,
        train.model_config.slm.sr,
    ).to(train.config.training.device)
    train.optimizer.add_discriminator_schedulers(train.discriminator_loss)

    # train.gl = MyDataParallel(train.gl)
    # train.dl = MyDataParallel(train.dl)
    # train.wl = MyDataParallel(train.wl)

    # TODO: How to access model diffusion?
    train.diffusion_sampler = DiffusionSampler(
        kdiffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )

    for k in train.optimizer.optimizers.keys():
        train.optimizer.optimizers[k] = train.accelerator.prepare(
            train.optimizer.optimizers[k]
        )
        train.optimizer.schedulers[k] = train.accelerator.prepare(
            train.optimizer.schedulers[k]
        )

    # adjust BERT learning rate
    # for g in train.optimizer.optimizers["bert"].param_groups:
    #    g["betas"] = (0.9, 0.99)
    #    g["lr"] = train.config.optimizer.bert_lr
    #    g["initial_lr"] = train.config.optimizer.bert_lr
    #    g["min_lr"] = 0
    #    g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    # for module in ["decoder", "style_encoder"]:
    #    for g in train.optimizer.optimizers[module].param_groups:
    #        g["betas"] = (0.0, 0.99)
    #        g["lr"] = train.config.optimizer.ft_lr
    #        g["initial_lr"] = train.config.optimizer.ft_lr
    #        g["min_lr"] = 0
    #        g["weight_decay"] = 1e-4

    train.n_down = 1  # TODO: Use train.model.text_aligner.n_down

    train.manifest.best_loss = float("inf")  # best test loss

    torch.cuda.empty_cache()

    train.stft_loss = MultiResolutionSTFTLoss().to(train.config.training.device)

    # logger.debug(f"BERT {optimizer.optimizers['bert']}")
    # logger.debug(f"decoder {optimizer.optimizers['decoder']}")

    train.start_ds = False

    # TODO: This value is calculated inconsistently based on whether checkpoints are loaded/saved
    train.manifest.running_std = []

    train.slm_adversarial_loss = SLMAdversarialLoss(
        train.model,
        train.wavlm_loss,
        train.diffusion_sampler,
        train.model_config.slmadv_params.min_len,
        train.model_config.slmadv_params.max_len,
        batch_percentage=train.model_config.slmadv_params.batch_percentage,
        skip_update=train.model_config.slmadv_params.iter,
        sig=train.model_config.slmadv_params.sig,
    )

    # for model in train.model:
    #    if model not in {"diffusion"}:
    # train.model.text_encoder.to("cuda")
    # train.model.text_encoder.lstm.flatten_parameters()
    # safetensors.torch.save_file(train.model.text_encoder.state_dict(), "tmp.safetensors")
    # train.accelerator.save_state("tmp")
    # exit()

    train_val_loop(train)
    train.accelerator.end_training()


def train_val_loop(train: TrainContext):
    if train.manifest.stage in {"first", "first_tma"}:
        train.train_batch = train_first
        train.validate = validate_first
    elif train.manifest.stage in {"vocoder"}:
        train.train_batch = train_vocoder_adapter
        train.validate = validate_first
    elif train.manifest.stage in {"acoustic"}:
        train.train_batch = train_acoustic_adapter
        train.validate = validate_first
    elif train.manifest.stage in {"second", "second_style", "second_joint"}:
        train.train_batch = train_second
        train.validate = validate_second
    else:
        exit(
            "Invalid training stage. --stage must be one of: 'first', 'first_tma', 'second', 'second_style', 'second_joint', 'acoustic', 'vocoder'"
        )
    while train.manifest.current_epoch <= train.manifest.max_epoch:
        train.batch_manager.init_epoch(train)
        train.manifest.steps_per_epoch = train.batch_manager.get_step_count()
        train.running_loss = 0
        train.start_time = time.time()

        # TODO: fix this logic if we plan on running a single file with multiple stages
        if train.manifest.stage == "second_style" or train.early_joint:
            train.start_ds = True

        # TODO: This line should be obsolete soon
        _ = [train.model[key].train() for key in train.model]
        for _, batch in enumerate(train.batch_manager.loader):
            train_val_iterate(batch, train)
        train.manifest.current_epoch += 1
        train.manifest.current_step = 0
        # TODO: change stages based on current epoch?
        train.manifest.training_log.append(
            f"Completed 1 epoch of {train.manifest.stage} training"
        )


def train_val_iterate(batch, train: TrainContext):
    train.batch_manager.train_iterate(batch, train)
    train.manifest.current_total_step += 1
    train.manifest.current_step += 1
    train.manifest.total_trained_audio_seconds += (
        float(len(batch[0][0]) * len(batch[0]))
        / train.model_config.preprocess.sample_rate
    )
    # filenames = batch[8]
    # logger.info(f"Step {train.manifest.current_step} Processing: {filenames}")
    num = train.manifest.current_total_step
    do_val = num % train.config.training.val_interval == 0
    do_save = num % train.config.training.save_interval == 0
    if do_val or do_save:
        train.validate(current_step=num, save=do_save, train=train)


if __name__ == "__main__":
    main()
