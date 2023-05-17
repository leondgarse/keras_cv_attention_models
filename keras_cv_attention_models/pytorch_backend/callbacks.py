class Callback:
    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, cur_epoch, logs=None):
        pass

    def on_epoch_end(self, cur_epoch, logs=None):
        pass


class Accuracy(Callback):
    def __init__(self):
        self.name = "acc"
        super().__init__()

    def on_epoch_begin(self, batch, logs=None):
        self.mean_acc, self.passed_steps = 0.0, 0

    def on_train_batch_end(self, batch, logs=None):
        self.mean_acc = (self.mean_acc * self.passed_steps + logs.get("accuracy", 0)) / (self.passed_steps + 1)
        self.passed_steps += 1
        return self.mean_acc
