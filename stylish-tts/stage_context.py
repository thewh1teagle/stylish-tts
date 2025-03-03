import traceback
from typing import Callable, List, Any
import torch
from munch import Munch

from stages import (
    train_first,
    validate_first,
    train_second,
    validate_second,
    train_acoustic_adapter,
    train_vocoder_adapter,
)

from loss_log import combine_logs
from stage_train import train_pre_acoustic, train_acoustic
from stage_validate import validate_acoustic
from optimizers import build_optimizer
from utils import get_image


class StageConfig:
    def __init__(
        self,
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        disc_models: List[str],
        inputs: List[str],
    ):
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.disc_models: List[str] = disc_models
        self.inputs: List[str] = inputs


stages = {
    "first": StageConfig(
        train_fn=train_first,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
    "first_tma": StageConfig(
        train_fn=train_first,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
    "second": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
    "second_style": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
    "second_joint": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
    "pre_acoustic": StageConfig(
        train_fn=train_pre_acoustic,
        validate_fn=validate_acoustic,
        train_models=["text_encoder", "style_encoder", "decoder"],
        eval_models=["text_aligner"],
        disc_models=[],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "sentence_embedding",
        ],
    ),
    "acoustic": StageConfig(
        train_fn=train_acoustic,
        validate_fn=validate_acoustic,
        train_models=["text_encoder", "style_encoder", "decoder", "text_aligner"],
        eval_models=[],
        disc_models=["msd", "mpd"],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "log_amplitude",
            "phase",
            "real",
            "imaginary",
            "sentence_emebdding",
        ],
    ),
    "vocoder": StageConfig(
        train_fn=train_vocoder_adapter,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        disc_models=[],
        inputs=[],
    ),
}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class StageContext:
    def __init__(self) -> None:
        self.max_epoch: int = 0
        self.steps_per_epoch: int = 0

        self.name = None
        self.train_fn = None
        self.validate_fn = None
        self.optimizer = None

    def begin_stage(self, name, train):
        if self.name is not None:
            self.optimizer.reset_lr(train)
        #    for key in train.model:
        #        train.model[key] = train.accelerator.free_memory(train.model[key])
        #    for key in train.model:
        #        train.model[key] = train.accelerator.prepare(train.model[key])
        #    self.optimizer.free_memory(train.accelerator)
        self.max_epoch = train.config.training_plan.dict()[name]
        self.steps_per_epoch = train.batch_manager.get_step_count()
        self.name = name
        self.train_fn = stages[name].train_fn
        self.validate_fn = stages[name].validate_fn
        if self.optimizer is None:
            self.optimizer = build_optimizer(
                self.max_epoch, self.steps_per_epoch, train=train
            )
            self.optimizer.prepare(train.accelerator)

    def train_batch(self, inputs, train):
        config = stages[self.name]
        batch = prepare_batch(inputs, train.config.training.device, config.inputs)
        model = prepare_model(
            train.model,
            train.config.training.device,
            config.train_models,
            config.eval_models,
            config.disc_models,
        )
        result = self.train_fn(batch, model, train)
        optimizer_step(self.optimizer, config.train_models)
        return result

    def validate(self, train):
        sample_count = 6
        for key in train.model:
            train.model[key].eval()
        logs = []
        for index, inputs in enumerate(train.val_dataloader):
            try:
                batch = prepare_batch(
                    inputs, train.config.training.device, stages[self.name].inputs
                )
                next_log, attention, audio_out, audio_gt = self.validate_fn(
                    batch, train
                )
                logs.append(next_log)
                if index < sample_count and train.accelerator.is_main_process:
                    attention = attention.cpu().numpy().squeeze()
                    audio_out = audio_out.cpu().numpy().squeeze()
                    audio_gt = audio_gt.cpu().numpy().squeeze()
                    steps = train.manifest.current_total_step
                    sample_rate = train.model_config.preprocess.sample_rate
                    train.writer.add_figure(
                        f"eval/attention_{index}", get_image(attention), steps
                    )
                    train.writer.add_audio(
                        f"eval/sample_{index}",
                        audio_out,
                        steps,
                        sample_rate=sample_rate,
                    )
                    train.writer.add_audio(
                        f"eval/sample_{index}_gt", audio_gt, 0, sample_rate=sample_rate
                    )

            except Exception as e:
                path = inputs[8]
                train.logger.error(f"Validation failed {path}: {e}")
                traceback.print_exc()
                continue
        validation = combine_logs(logs)
        validation.broadcast(train.manifest, train.stage, validation=True)
        total = validation.total()
        if total < train.manifest.best_loss:
            train.manifest.best_loss = total
        for key in train.model:
            train.model[key].train()


batch_names = [
    "audio_gt",
    "text",
    "text_length",
    "ref_text",
    "ref_length",
    "mel",
    "mel_length",
    "ref_mel",
    "path",
    "pitch",
    "log_amplitude",
    "phase",
    "real",
    "imaginary",
    "sentence_embedding",
]


def prepare_batch(
    inputs: List[Any], device: torch.device, keys_to_transfer: List[str]
) -> Munch:
    """
    Transfers selected batch elements to the specified device.
    """
    prepared = {}
    for i, key in enumerate(batch_names):
        if key in keys_to_transfer:
            if key != "paths":
                prepared[key] = inputs[i].to(device)
            else:
                prepared[key] = inputs[i]
    return Munch(**prepared)


def prepare_model(model, device, training_set, eval_set, disc_set) -> Munch:
    """
    Prepares models for training or evaluation, attaches them to the cpu memory if unused, returns an object which contains only the models that will be used.
    """
    result = {}
    for key in model:
        if key in training_set or key in eval_set or key in disc_set:
            result[key] = model[key]
            result[key].to(device)
        else:
            model[key].to("cpu")
        # if key in training_set or key in disc_set:
        #    result[key].train()
        # elif key in eval_set:
        #    result[key].eval()
    return Munch(**result)


def optimizer_step(optimizer, keys: List[str]) -> None:
    """
    Steps the optimizer for each module key in keys.
    """
    for key in keys:
        optimizer.step(key)
