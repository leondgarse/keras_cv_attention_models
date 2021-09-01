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


def constant_scheduler(epoch, lr_base, lr_decay_steps, decay_rate=0.1, lr_min=0, warmup=0):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        epoch -= warmup
        lr = lr_base * decay_rate ** np.sum(epoch >= np.array(lr_decay_steps))
        lr = lr if lr > lr_min else lr_min
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
        if hasattr(lr, 'value'):
            lr = lr.value()

        self.history.setdefault("lr", []).append(float(lr))
        for k, v in logs.items():
            k = "accuracy" if "accuracy" in k else k
            self.history.setdefault(k, []).append(float(v))
        if len(self.model.losses) != 0:
            regular_loss = K.sum(self.model.losses).numpy()
            self.history.setdefault("regular_loss", []).append(float(regular_loss))
            self.history["loss"][-1] -= regular_loss

        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        print("{")
        for kk, vv in self.history.items():
            print("  '%s': %s," % (kk, vv))
        print("}")
