from config_loader import Config
from meldataset import BatchManager


class Manifest:
    def __init__(self):
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.iters: int = 0
        self.stage: str = "first"
        self.epochs: int = 0
        self.training_log: str = []


class TrainContext:
    def __init__(self):
        self.config: Config = None
        self.batch_manager: BatchManager = None
        self.manifest: Manifest = Manifest()
        pass
