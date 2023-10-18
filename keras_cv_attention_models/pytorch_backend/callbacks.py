import sys
import numpy as np


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

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass


class TerminateOnNaN(Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if not np.isfinite(loss):
                print("\nError: Invalid loss, terminating training")
                # self.model.stop_training = True
                sys.exit()


class TensorBoard(Callback):
    def __init__(self, log_dir="logs", histogram_freq=1, **kwargs):
        super().__init__()
        self.log_dir, self.histogram_freq = log_dir, histogram_freq
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard_writer = SummaryWriter(self.log_dir)
            print(">>>> Tensorboard writer created, summary will be write to '{}', view by 'tensorboard --logdir {}'".format(log_dir, log_dir))
        except:
            self.tensorboard_writer = None
            print("[Error] tensorboard not installed, try `pip install tensorboard`")

    def on_epoch_end(self, cur_epoch, logs=None):
        if self.tensorboard_writer is None:
            return
        logs = logs or {}
        for kk, vv in logs.items():
            self.tensorboard_writer.add_scalar(kk, vv, cur_epoch)
