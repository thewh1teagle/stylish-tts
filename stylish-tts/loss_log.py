from train_context import TrainContext
import logging

logger = logging.getLogger(__name__)


class LossLog:
    def __init__(self, logger, writer, loss_weights):
        self.logger = logger
        self.writer = writer
        self.weights = loss_weights
        self.metrics = {}
        self.total_loss = None

    def total(self):
        if self.total_loss is None:
            self.calculate_metrics()
        return self.total_loss

    def broadcast(self, manifest, stage):
        if self.total_loss is None:
            self.calculate_metrics()
        self.logger.info(
            f"Epoch [{manifest.current_epoch}/{stage.max_epoch}], Step [{manifest.current_step}/{stage.steps_per_epoch}], loss: {self.total_loss:.3f}, "
            + ", ".join(f"{k}: {v:.3f}" for k, v in self.metrics.items())
        )
        self.writer.add_scalar(
            "train/loss", self.total_loss, manifest.current_total_step
        )
        for key, value in self.metrics.items():
            self.writer.add_scalar(f"train/{key}", value, manifest.current_total_step)

    def weight(self, key: str):
        if key in self.weights:
            return self.weights[key]
        else:
            # logging.error(f"WARNING: Unknown weight for key {key}, defaulting to 1")
            logging.debug(f"self.weights: {self.weights}")
            return 1

    def update_loss(self, key, value, count):
        return weight

    def calculate_metrics(self):
        total = 0
        total_weight = 0
        for key, value in self.metrics.items():
            weight = self.weight(key)
            loss = value * weight
            total += loss
            total_weight += weight
        self.total_loss = total / total_weight

    def add_loss(self, key, value):
        self.metrics[key] = value


def combine_logs(loglist):
    result = None
    if len(loglist) > 0:
        result = LossLog(loglist[0].logger, loglist[0].writer, loglist[0].weights)
        for log in loglist:
            for key in log.metrics.keys():
                if key not in result.metrics:
                    result.metrics[key] = 0
                result.metrics[key] += log.metrics[key]
    return result
