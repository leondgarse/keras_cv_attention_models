class Callback:
    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

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

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass


class Accuracy(Callback):
    def __init__(self):
        self.name = "acc"
        super().__init__()
        self.on_epoch_begin()

    def on_epoch_begin(self, batch=0, logs=None):
        self.sum_value, self.passed_steps = 0.0, 0

    def on_train_batch_end(self, batch=0, logs=None):
        self.sum_value = self.sum_value + logs.get("accuracy", 0)
        self.passed_steps += 1
        return self.sum_value / self.passed_steps
