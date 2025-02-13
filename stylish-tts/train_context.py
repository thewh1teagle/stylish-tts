from config_loader import Config
from batch_manager import BatchManager
from typing import Callable


class Manifest:
    def __init__(self):
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.steps_per_epoch: int = 0
        self.current_total_step: int = 0
        self.total_trained_audio_seconds: float = 0
        self.stage: str = "first"
        self.max_epoch: int = 0
        self.training_log: str = []


class TrainContext:
    def __init__(self):
        self.config: Config = None
        self.batch_manager: BatchManager = None
        self.manifest: Manifest = Manifest()
        self.validate: Callable = None
        self.train_batch: Callable = None
        pass
