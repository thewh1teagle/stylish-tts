from stages import (
    train_first,
    validate_first,
    train_second,
    validate_second,
    train_acoustic_adapter,
    train_vocoder_adapter,
)
from optimizers import build_optimizer

stages = {
    "first": {
        "train": train_first,
        "validate": validate_first,
    },
    "first_tma": {
        "train": train_first,
        "validate": validate_first,
    },
    "second": {
        "train": train_second,
        "validate": validate_second,
    },
    "second_style": {
        "train": train_second,
        "validate": validate_second,
    },
    "second_joint": {
        "train": train_second,
        "validate": validate_second,
    },
    "acoustic": {
        "train": train_acoustic_adapter,
        "validate": validate_first,
    },
    "vocoder": {
        "train": train_vocoder_adapter,
        "validate": validate_first,
    },
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
        self.train_batch = None
        self.validate = None
        self.optimizer = None

    def begin_stage(self, name, train):
        if self.name is not None:
            self.optimizer.free_memory(train.accelerator)
        is_second = name in {"second", "second_style", "second_joint"}
        self.max_epoch = train.config.training_plan.dict()[name]
        self.steps_per_epoch = train.batch_manager.get_step_count()
        self.name = name
        self.train_batch = stages[name]["train"]
        self.validate = stages[name]["validate"]
        self.optimizer = build_optimizer(
            self.max_epoch, self.steps_per_epoch, is_second=is_second, train=train
        )
        self.optimizer.prepare(train.accelerator)
