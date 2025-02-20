# coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW
import logging
import transformers

logger = logging.getLogger(__name__)


class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        # self.scale_schedulers = {}
        self.disc_schedulers = {}
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(
            lambda x, y: x + y, [v.param_groups for v in self.optimizers.values()]
        )
        # for key in self.keys:
        #    self.scale_schedulers[key] = ScaleLR(self.optimizers[key])

    def prepare(self, accelerator):
        for key in self.optimizers.keys():
            self.optimizers[key] = accelerator.prepare(self.optimizers[key])
            self.schedulers[key] = accelerator.prepare(self.schedulers[key])
        # for key in self.disc_schedulers.keys():
        #    self.disc_schedulers[key] = accelerator.prepare(self.disc_schedulers[key])

    def free_memory(self, accelerator):
        for key in self.optimizers.keys():
            accelerator.free_memory(self.optimizers[key])
            accelerator.free_memory(self.schedulers[key])
        # for key in self.disc_schedulers.keys():
        #    accelerator.free_memory(self.disc_schedulers[key])

    def add_discriminator_schedulers(self, discriminator_loss):
        for key in ["msd", "mpd"]:
            self.disc_schedulers[key] = torch.optim.lr_scheduler.MultiplicativeLR(
                self.optimizers[key], discriminator_loss.get_disc_lambda()
            )

    def step_discriminator_schedulers(self):
        for key in ["msd", "mpd"]:
            self.disc_schedulers[key].step()

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict()) for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                logger.info("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

    # def scale(self, scale, key_in=None):
    #    keys = [key_in]
    #    if key_in is None:
    #        keys = self.keys
    #    for key in keys:
    #        self.scale_schedulers[key].set_factor(scale)
    #        self.scale_schedulers[key].step()


class ScaleLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0,
    ):
        self.factor = factor
        super().__init__(optimizer)

    def set_factor(self, factor):
        self.factor = factor

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        return [group["lr"] * self.factor for group in self.optimizer.param_groups]


def build_optimizer(max_epoch, steps_per_epoch, is_second=False, train=None):
    optim = {}
    schedulers = {}
    for key in train.model.keys():
        lr = train.config.optimizer.lr
        weight_decay = 1e-4
        betas = (0.0, 0.99)
        if is_second:
            if key == "bert":
                lr = train.config.optimizer.bert_lr
                weight_decay = 1e-2
                betas = (0.9, 0.99)
            elif key in {"decoder", "style_encoder"}:
                lr = train.config.optimizer.ft_lr
        optim[key] = AdamW(
            train.model[key].parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-9,
        )
        schedulers[key] = transformers.get_cosine_schedule_with_warmup(
            optim[key],
            num_warmup_steps=200,
            num_training_steps=steps_per_epoch * max_epoch,
        )

    multi_optim = MultiOptimizer(optim, schedulers)
    multi_optim.add_discriminator_schedulers(train.discriminator_loss)
    return multi_optim
