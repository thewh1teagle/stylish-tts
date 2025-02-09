from train_context import TrainContext


class LossLog:
    def __init__(self, train: TrainContext):
        self.logger = train.logger
        self.writer = train.writer
        self.weights = train.config.loss_weights
        self.count = 0
        self.incremental = {}
        self.single = {}
        self.metrics = None
        self.total_loss = 0

    def reset(self):
        self.count = 0
        self.incremental = {}
        self.single = {}
        self.metrics = None
        self.total_loss = 0

    def total(self):
        if self.metrics is None:
            self.calculate_results()
        return results["total_loss"]

    def broadcast(self, manifest):
        if self.metrics is None:
            self.calculate_metrics()
        self.logger.info(
            f"Epoch [{manifest.current_epoch}/{manifest.epochs}], Step [TODO:STEPCOUNT], loss: {self.total_loss}, "
            + ", ".join(f"{k}: {v:.5f}" for k, v in self.metrics.items())
        )
        train.writer.add_scalar("train/loss", self.total_loss, manifest.iters)
        for key, value in self.metrics.items():
            train.writer.add_scalar(f"train/{key}", value, manifest.iters)

    def weight(self, key: str):
        if key in self.weights:
            return self.weights[key]
        else:
            self.logger(f"WARNING: Unknown weight for key {key}, defaulting to 1")
            return 1

    def update_loss(self, key, value, count):
        weight = self.weight(key)
        loss = value * weight / count
        self.total_loss += loss
        self.metrics[key] = loss
        return weight

    def calculate_metrics(self):
        self.metrics = {}
        self.total_loss = 0
        total_weight = 0
        if count > 0:
            for key, value in self.incremental.items():
                total_weight += self.update_loss(key, value, self.count)
        for key, value in self.single.items():
            total_weight += self.update_loss(key, value, 1)
        self.total_loss = self.total_loss / weight_total

    def add_loss(self, key, value, incremental=False):
        if incremental:
            if key not in self.incremental:
                self.incremental[key] = 0
            self.incremental[key] += value
            self._add_loss(self.incremental)
        else:
            self.single[key] = value

    def increment_loss_count(self):
        self.count += 1
