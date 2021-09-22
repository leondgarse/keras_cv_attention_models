import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, first_restart_step, steps_per_epoch=-1, m_mul=0.5, t_mul=2.0, lr_min=1e-5, warmup=0):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.m_mul, self.t_mul, self.lr_min = lr_base, m_mul, t_mul, lr_min
        self.first_restart_step, self.warmup, self.steps_per_epoch = first_restart_step, warmup, steps_per_epoch
        self.init_step_num, self.cur_epoch = 0, 0
        if steps_per_epoch != -1:
            self.build(steps_per_epoch)

    def build(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.warmup *= steps_per_epoch
        self.first_restart_step *= steps_per_epoch
        alpha = self.lr_min / self.lr_base
        if self.lr_min == self.lr_base * self.m_mul:  # Without restart
            self.schedule = keras.experimental.CosineDecay(self.lr_base, self.first_restart_step, alpha=alpha)
        else:
            self.schedule = keras.experimental.CosineDecayRestarts(self.lr_base, self.first_restart_step, t_mul=self.t_mul, m_mul=self.m_mul, alpha=alpha)

        if self.warmup != 0:
            self.warmup_lr_func = lambda ii: self.lr_min + (self.lr_base - self.lr_min) * ii / self.warmup

    def on_epoch_begin(self, cur_epoch, logs=None):
        self.init_step_num = int(self.steps_per_epoch * cur_epoch)
        self.cur_epoch = cur_epoch

    def on_train_batch_begin(self, iterNum, logs=None):
        global_iterNum = iterNum + self.init_step_num
        if global_iterNum < self.warmup:
            lr = self.warmup_lr_func(global_iterNum)
        else:
            lr = self.schedule(global_iterNum - self.warmup)

        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        if iterNum == 0:
            print("\nLearning rate for iter {} is {}".format(self.cur_epoch + 1, lr))
        return lr


class CosineLrSchedulerEpoch(keras.callbacks.Callback):
    def __init__(self, lr_base, first_restart_step, m_mul=0.5, t_mul=2.0, lr_min=1e-5, warmup=0):
        super(CosineLrSchedulerEpoch, self).__init__()
        self.warmup = warmup

        if lr_min == lr_base * m_mul:
            self.schedule = keras.experimental.CosineDecay(lr_base, first_restart_step, alpha=lr_min / lr_base)
        else:
            self.schedule = keras.experimental.CosineDecayRestarts(
                lr_base, first_restart_step, t_mul=t_mul, m_mul=m_mul, alpha=lr_min / lr_base
            )

        if warmup != 0:
            self.warmup_lr_func = lambda ii: lr_min + (lr_base - lr_min) * ii / warmup

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            lr = self.warmup_lr_func(epoch)
        else:
            lr = self.schedule(epoch - self.warmup)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)

        print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
        return lr


def constant_scheduler(epoch, lr_base, lr_decay_steps, decay_rate=0.1, warmup=0):
    if epoch < warmup:
        lr = lr_base * (epoch + 1) / (warmup + 1)
    else:
        epoch -= warmup
        lr = lr_base * decay_rate ** np.sum(epoch >= np.array(lr_decay_steps))
    print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def exp_scheduler(epoch, lr_base=0.1, decay_step=1, decay_rate=0.9, lr_min=0, warmup=0):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        epoch -= warmup
        lr = lr_base * decay_rate ** (epoch / decay_step)
        lr = lr if lr > lr_min else lr_min
    # print("Learning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


class OptimizerWeightDecay(keras.callbacks.Callback):
    def __init__(self, lr_base, wd_base, is_lr_on_batch=False):
        super(OptimizerWeightDecay, self).__init__()
        self.wd_m = wd_base / lr_base
        self.lr_base, self.wd_base = lr_base, wd_base
        # self.model.optimizer.weight_decay = lambda: wd_m * self.model.optimizer.lr
        self.is_lr_on_batch = is_lr_on_batch
        if is_lr_on_batch:
            self.on_train_batch_begin = self.__update_wd__
        else:
            self.on_epoch_begin = self.__update_wd__

    def __update_wd__(self, step, log=None):
        if self.model is not None:
            wd = self.wd_m * K.get_value(self.model.optimizer.lr)
            # wd = self.wd_base * K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.weight_decay, wd)
        # wd = self.model.optimizer.weight_decay
        if not self.is_lr_on_batch or step == 0:
            print("Weight decay is {}".format(wd))


class MyHistory(keras.callbacks.Callback):
    def __init__(self, initial_file=None):
        super(MyHistory, self).__init__()
        if initial_file and os.path.exists(initial_file):
            with open(initial_file, "r") as ff:
                self.history = json.load(ff)
        else:
            self.history = {}
        self.initial_file = initial_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.pop("lr", None)
        lr = self.model.optimizer.lr
        if hasattr(lr, "value"):
            lr = lr.value()

        self.history.setdefault("lr", []).append(float(lr))
        for k, v in logs.items():
            k = "accuracy" if "accuracy" in k else k
            self.history.setdefault(k, []).append(float(v))

        if len(self.model.losses) != 0: # Has regular_loss
            regular_loss = K.sum(self.model.losses).numpy()
            self.history.setdefault("regular_loss", []).append(float(regular_loss))
            self.history["loss"][-1] -= regular_loss
            if "val_loss" in self.history:
                self.history["val_loss"][-1] -= regular_loss

        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        print("{")
        for kk, vv in self.history.items():
            print("  '%s': %s," % (kk, vv))
        print("}")


class MyCheckpoint(keras.callbacks.Callback):
    def __init__(self, basic_save_name, monitor="val_acc", mode="auto", save_path="checkpoints"):
        super(MyCheckpoint, self).__init__()
        self.monitor_save = os.path.join(save_path, basic_save_name + "_epoch_{}_" + monitor + "_{}.h5")
        self.monitor_save_re = self.monitor_save.format("*", "*")
        self.latest_save = os.path.join(save_path, basic_save_name + "_latest.h5")
        self.monitor = monitor
        self.is_better = (lambda cur, pre: cur <= pre) if mode == "min" or "loss" in monitor else (lambda cur, pre: cur >= pre)
        self.pre_best = 1e5 if mode == "min" or "loss" in monitor else -1e5

    def on_epoch_end(self, epoch, logs={}):
        # tf.print(">>>> Save latest to:", self.latest_save)
        self.model.save(self.latest_save)

        cur_monitor_val = logs.get(self.monitor, 0)
        if self.is_better(cur_monitor_val, self.pre_best):
            self.pre_best = cur_monitor_val
            pre_monitor_saves = tf.io.gfile.glob(self.monitor_save_re)
            # tf.print(">>>> pre_monitor_saves:", pre_monitor_saves)
            if len(pre_monitor_saves) != 0:
                os.remove(pre_monitor_saves[0])
            monitor_save = self.monitor_save.format(epoch, "{:.4f}".format(cur_monitor_val))
            tf.print(">>>> Save best to:", monitor_save)
            self.model.save(monitor_save)
