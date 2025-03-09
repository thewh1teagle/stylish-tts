# coding:utf-8
import torch
from torch.optim import AdamW
import logging
import transformers

logger = logging.getLogger(__name__)
logical_step_limit = 10000
logical_step_warmup = 100


class MultiOptimizer:
    def __init__(
        self, optimizers={}, schedulers={}, min_disc_lr=1e-5, max_disc_lr=1e-3
    ):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.min_disc_lr = min_disc_lr
        self.max_disc_lr = max_disc_lr
        # self.scale_schedulers = {}
        self.disc_schedulers = {}
        self.keys = list(optimizers.keys())

    def prepare(self, accelerator):
        for key in self.optimizers.keys():
            self.optimizers[key] = accelerator.prepare(self.optimizers[key])
            self.schedulers[key] = accelerator.prepare(self.schedulers[key])

    def reset_lr(self, stage_name, train):
        for key in train.model.keys():
            lr, _, _ = calculate_lr(key, stage_name, train=train)
            for param_group in self.optimizers[key].param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(lr)
                    param_group["initial_lr"].fill_(lr)
                else:
                    param_group["lr"] = lr
                    param_group["initial_lr"] = lr
            self.schedulers[key].scheduler.last_epoch = -1
            self.schedulers[key].step()
        self.reset_discriminator_schedulers()

    def add_discriminator_schedulers(self, discriminator_loss):
        for key in ["msd", "mpd"]:
            self.disc_schedulers[key] = torch.optim.lr_scheduler.MultiplicativeLR(
                self.optimizers[key], discriminator_loss.get_disc_lambda()
            )

    def step_discriminator_schedulers(self):
        for key in ["msd", "mpd"]:
            self.disc_schedulers[key].step()
            for param_group in self.optimizers[key].param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    torch.clamp(
                        param_group["lr"],
                        min=self.min_disc_lr,
                        max=self.max_disc_lr,
                        out=param_group["lr"],
                    )
                else:
                    param_group["lr"] = min(
                        max(self.min_disc_lr, param_group["lr"]), self.max_disc_lr
                    )

    def reset_discriminator_schedulers(self):
        for key in ["msd", "mpd"]:
            for param_group in self.optimizers[key].param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"] = zeros_like(
                        param_group["lr"], device=param_group["lr"].device
                    )
                else:
                    param_group["lr"] = 0

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

    def scheduler(self, step: int, step_limit: int):
        logical_step = step * logical_step_limit // step_limit
        for key in self.keys:
            self.schedulers[key].scheduler.last_epoch = logical_step
            self.schedulers[key].step()


def build_optimizer(stage_name: str, *, train):
    optim = {}
    schedulers = {}
    for key in train.model.keys():
        lr, weight_decay, betas = calculate_lr(key, stage_name, train=train)
        optim[key] = AdamW(
            train.model[key].parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-9,
        )
        schedulers[key] = transformers.get_cosine_schedule_with_warmup(
            optim[key],
            num_warmup_steps=logical_step_warmup,
            num_training_steps=logical_step_limit,
        )
    min_disc_lr = train.config.optimizer.lr / 10
    max_disc_lr = train.config.optimizer.lr * 10
    multi_optim = MultiOptimizer(optim, schedulers, min_disc_lr, max_disc_lr)
    multi_optim.add_discriminator_schedulers(train.discriminator_loss)
    return multi_optim


def calculate_lr(key, stage_name, *, train):
    is_second = (
        stage_name == "second"
        or stage_name == "second_style"
        or stage_name == "second_joint"
        or stage_name == "textual"
    )
    lr = train.config.optimizer.lr
    weight_decay = 1e-4
    betas = (0.85, 0.99)
    if is_second:
        if key == "bert":
            lr = train.config.optimizer.bert_lr
            weight_decay = 1e-2
            betas = (0.9, 0.99)
        elif key in {"decoder", "style_encoder"}:
            lr = train.config.optimizer.ft_lr
    return lr, weight_decay, betas
