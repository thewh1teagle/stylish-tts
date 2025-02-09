from train_context import TrainContext


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

    def broadcast(self, manifest):
        if self.total_loss is None:
            self.calculate_metrics()
        self.logger.info(
            f"Epoch [{manifest.current_epoch}/{manifest.epochs}], Step [TODO:STEPCOUNT], loss: {self.total_loss}, "
            + ", ".join(f"{k}: {v:.5f}" for k, v in self.metrics.items())
        )
        self.writer.add_scalar("train/loss", self.total_loss, manifest.iters)
        for key, value in self.metrics.items():
            self.writer.add_scalar(f"train/{key}", value, manifest.iters)

    def weight(self, key: str):
        if key in self.weights:
            return self.weights[key]
        else:
            print(f"WARNING: Unknown weight for key {key}, defaulting to 1")
            print(self.weights)
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
