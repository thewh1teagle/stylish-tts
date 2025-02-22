from typing import Callable

from stages import (
    train_first,
    validate_first,
    train_second,
    validate_second,
    train_acoustic_adapter,
    train_vocoder_adapter,
)
from optimizers import build_optimizer


class StageConfig:
    def __init__(
        self,
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        inputs: List[str],
    ):
        self.train_fn: Callabale = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.inputs: List[str] = inputs


stages = {
    "first": StageConfig(
        train_fn=train_first,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "first_tma": StageConfig(
        train_fn=train_first,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "second": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "second_style": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "second_joint": StageConfig(
        train_fn=train_second,
        validate_fn=validate_second,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "acoustic": StageConfig(
        train_fn=train_acoustic_adapter,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
        inputs=[],
    ),
    "vocoder": StageConfig(
        train_fn=train_vocoder_adapter,
        validate_fn=validate_first,
        train_models=[],
        eval_models=[],
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
            self.optimizer.free_memory(train.accelerator)
        is_second = name in {"second", "second_style", "second_joint"}
        self.max_epoch = train.config.training_plan.dict()[name]
        self.steps_per_epoch = train.batch_manager.get_step_count()
        self.name = name
        self.train_fn = stages[name].train_fn
        self.validate_fn = stages[name].validate_fn
        self.optimizer = build_optimizer(
            self.max_epoch, self.steps_per_epoch, is_second=is_second, train=train
        )
        self.optimizer.prepare(train.accelerator)

    def train_batch(self, *args):
        return self.train_fn(*args)

    def validate(self, *args):
        return self.validate(*args)
