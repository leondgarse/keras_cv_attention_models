#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.model_surgery import change_model_input_shape
import tensorflow as tf

class Torch_model_interf:
    def __init__(self, model):
        import torch
        import os

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        self.model = model.cuda(device_name)

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        output = self.model(self.torch.from_numpy(imgs).permute([0, 3, 1, 2]).to(self.device).float())
        return output.cpu().detach().numpy()



def eval(model, data_name="imagenet2012", input_shape=None, batch_size=64, central_crop=1.0, resize_method="bicubic", rescale_mode="torch"):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
    _, test_dataset, _, _, _ = data.init_dataset(
        data_name, input_shape=input_shape, batch_size=batch_size, central_crop=central_crop, resize_method=resize_method, rescale_mode=rescale_mode
    )

    model_interf = change_model_input_shape(model, input_shape) if isinstance(model, tf.keras.models.Model) else Torch_model_interf(model)

    y_true, y_pred_top_1, y_pred_top_5 = [], [], []
    for img_batch, true_labels in tqdm(test_dataset.as_numpy_iterator(), "Evaluating", total=len(test_dataset)):
        predicts = np.array(model_interf(img_batch))
        pred_argsort = predicts.argsort(-1)
        y_pred_top_1.extend(pred_argsort[:, -1])
        y_pred_top_5.extend(pred_argsort[:, -5:])
        y_true.extend(np.array(true_labels).argmax(-1))
    y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
    accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
    accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
    print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
    return y_true, y_pred_top_1, y_pred_top_5


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
    eval(model, args.data_name, input_shape, args.batch_size, args.central_crop, args.resize_method, args.rescale_mode)
