#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from keras_cv_attention_models.imagenet import evaluation
from keras_cv_attention_models.model_surgery import change_model_input_shape
import tensorflow as tf


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Saved h5 model path")
    parser.add_argument("-i", "--input_shape", type=int, default=-1, help="Model input shape, Set -1 for using model.input_shape")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--data_name", type=str, default="imagenet2012", help="Dataset name from tensorflow_datasets like imagenet2012 cifar10")
    parser.add_argument("--rescale_mode", type=str, default="torch", help="[Dataset] Rescale mode, one of [tf, torch]")
    parser.add_argument("--central_crop", type=float, default=1.0, help="[Dataset] Central crop fraction. Set 1 to disable")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="[Dataset] Resize method from tf.image.resize, like [bilinear, bicubic]")

    args = parser.parse_known_args(argv)[0]
    return args


if __name__ == "__main__":
    gpus = tf.config.experimental.get_visible_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    import tensorflow_addons as tfa
    import keras_cv_attention_models
    import sys

    args = parse_arguments(sys.argv[1:])

    model = tf.keras.models.load_model(args.model_path)
    input_shape = None if args.input_shape == -1 else (args.input_shape, args.input_shape)
    evaluation(model, args.data_name, input_shape, args.batch_size, args.central_crop, args.resize_method, args.rescale_mode)
