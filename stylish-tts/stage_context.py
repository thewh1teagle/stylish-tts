import json
import os
import os.path as osp
import traceback
from typing import Callable, List, Any, Dict, Optional
import torch
from munch import Munch
import tqdm

from loss_log import combine_logs
from stage_train import (
    train_pre_acoustic,
    train_acoustic,
    train_pre_textual,
    train_textual,
    train_joint,
)
from stage_validate import validate_acoustic, validate_textual
from optimizers import build_optimizer
from utils import get_image


class StageConfig:
    def __init__(
        self,
        next_stage: Optional[str],
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        disc_models: List[str],
        inputs: List[str],
    ):
        self.next_stage: Optional[str] = next_stage
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.disc_models: List[str] = disc_models
        self.inputs: List[str] = inputs


stages = {
    "pre_acoustic": StageConfig(
        next_stage="acoustic",
        train_fn=train_pre_acoustic,
        validate_fn=validate_acoustic,
        train_models=["text_encoder", "acoustic_style_encoder", "decoder"],
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
            "voiced",
        ],
    ),
    "acoustic": StageConfig(
        next_stage="pre_textual",
        train_fn=train_acoustic,
        validate_fn=validate_acoustic,
        train_models=[
            "text_encoder",
            "acoustic_style_encoder",
            "decoder",
            "text_aligner",
        ],
        eval_models=[],
        disc_models=["msd", "mpd"],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "sentence_embedding",
            "voiced",
        ],
    ),
    "pre_textual": StageConfig(
        next_stage="textual",
        train_fn=train_pre_textual,
        validate_fn=validate_textual,
        train_models=[
            "acoustic_prosody_encoder",
            "duration_predictor",
            "pitch_energy_predictor",
            "bert",
            "bert_encoder",
        ],
        eval_models=[
            "text_encoder",
            "acoustic_style_encoder",
            "decoder",
            "text_aligner",
        ],
        disc_models=[],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "sentence_embedding",
            "voiced",
        ],
    ),
    "textual": StageConfig(
        next_stage=None,
        train_fn=train_textual,
        validate_fn=validate_textual,
        train_models=[
            "acoustic_prosody_encoder",
            "duration_predictor",
            "pitch_energy_predictor",
            "bert",
            "bert_encoder",
        ],
        eval_models=[
            "text_encoder",
            "acoustic_style_encoder",
            "decoder",
            "text_aligner",
        ],
        disc_models=[],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "sentence_embedding",
            "voiced",
        ],
    ),
    "joint": StageConfig(
        next_stage=None,
        train_fn=train_joint,
        validate_fn=validate_textual,
        train_models=[
            "acoustic_prosody_encoder",
            "duration_predictor",
            "pitch_energy_predictor",
            "bert",
            "bert_encoder",
        ],
        eval_models=[
            "text_encoder",
            "decoder",
            "acoustic_style_encoder",
            "text_aligner",
        ],
        disc_models=["msd", "mpd"],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "sentence_embedding",
            "voiced",
        ],
    ),
}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class StageContext:
    def __init__(
        self, name: str, train, train_time_bins: dict, val_time_bins: dict
    ) -> None:
        self.name: str = name
        self.max_epoch: int = train.config.training_plan.dict()[name]
        self.train_time_bins: dict = train_time_bins
        self.val_time_bins: dict = val_time_bins

        self.batch_sizes: Dict[str, int] = {}
        self.last_batch_load = None
        self.out_dir = train.out_dir
        self.load_batch_sizes()
        train.manifest.steps_per_epoch = self.get_steps_per_epoch()

        self.train_fn: Callable = stages[name].train_fn
        self.validate_fn: Callable = stages[name].validate_fn
        self.optimizer = build_optimizer(self.name, train=train)
        self.optimizer.prepare(train.accelerator)

    def begin_stage(self, name, train):
        self.name = name
        self.max_epoch = train.config.training_plan.dict()[name]
        self.train_fn = stages[name].train_fn
        self.validate_fn = stages[name].validate_fn
        self.optimizer.reset_lr(name, train)
        train.reset_out_dir(name)
        self.last_batch_load = None
        self.out_dir = train.out_dir
        self.load_batch_sizes()

    def get_next_stage(self) -> Optional[str]:
        return stages[self.name].next_stage

    def set_batch_size(self, i: int, batch_size: int) -> None:
        self.batch_sizes[str(i)] = batch_size

    def get_batch_size(self, key: int) -> int:
        if str(key) in self.batch_sizes:
            return self.batch_sizes[str(key)]
        else:
            return 1

    def reset_batch_sizes(self) -> None:
        self.batch_sizes = {}

    def batch_sizes_exist(self):
        return self.last_batch_load is not None

    def load_batch_sizes(self) -> None:
        batch_file = osp.join(self.out_dir, f"{self.name}_batch_sizes.json")
        if osp.isfile(batch_file):
            modified = os.stat(batch_file).st_mtime
            if self.last_batch_load is None or modified > self.last_batch_load:
                with open(batch_file, "r") as batch_input:
                    self.batch_sizes = json.load(batch_input)
                    self.last_batch_load = modified

    def save_batch_sizes(self) -> None:
        batch_file = osp.join(self.out_dir, f"{self.name}_batch_sizes.json")
        with open(batch_file, "w") as o:
            json.dump(self.batch_sizes, o)

    def get_steps_per_val(self) -> int:
        return self.get_steps(self.val_time_bins)

    def get_steps_per_epoch(self) -> int:
        return self.get_steps(self.train_time_bins)

    def get_steps(self, time_bins):
        total = 0
        for key in time_bins.keys():
            val = time_bins[key]
            total_batch = self.get_batch_size(key)
            if total_batch > 0:
                total += len(val) // total_batch + 1
                # if not self.drop_last and len(val) % total_batch != 0:
                #     total += 1
        return total

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
        progress_bar = None
        if train.accelerator.is_main_process:
            iterator = tqdm.tqdm(
                iterable=enumerate(train.val_dataloader),
                desc=f"Validating {self.name}",
                total=self.get_steps_per_val(),
                unit="steps",
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {remaining}{postfix} ",
                colour="BLUE",
                delay=2,
                leave=False,
                dynamic_ncols=True,
            )
            progress_bar = iterator
        else:
            iterator = enumerate(train.val_dataloader)
        for index, inputs in iterator:
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
                    sample_rate = train.model_config.sample_rate
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
                if train.accelerator.is_main_process:
                    interim = combine_logs(logs)
                    if progress_bar is not None and interim is not None:
                        progress_bar.set_postfix({"loss": f"{interim.total():.3f}"})

            except Exception as e:
                path = inputs[8]
                progress_bar.clear() if progress_bar is not None else None
                train.logger.error(f"Validation failed {path}: {e}")
                traceback.print_exc()
                progress_bar.display() if progress_bar is not None else None
                continue
        progress_bar.close() if progress_bar is not None else None
        validation = combine_logs(logs)
        if validation is not None:
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
    "sentence_embedding",
    "voiced",
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
