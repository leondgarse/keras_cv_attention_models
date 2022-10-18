#!/usr/bin/env python3
import json
from keras_cv_attention_models.imagenet import evaluation
import tensorflow as tf


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Could be: 1. Saved h5 / tflite model path. 2. Model name defined in this repo, format [sub_dir].[model_name] like regnet.RegNetZD8. 3. timm model like timm.models.resmlp_12_224",
    )
    parser.add_argument("-i", "--input_shape", type=int, default=-1, help="Model input shape, Set -1 for using model.input_shape")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--data_name", type=str, default="imagenet2012", help="Dataset name from tensorflow_datasets like imagenet2012 cifar10")
    parser.add_argument(
        "--rescale_mode", type=str, default="auto", help="Rescale mode, one of [tf, torch, raw, raw01, tf128]. Default `auto` means using model preset"
    )
    parser.add_argument("--central_crop", type=float, default=0.95, help="Central crop fraction. Set -1 to disable")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    parser.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    parser.add_argument("--num_classes", type=int, default=None, help="num_classes if not inited from h5 file. None for model.num_classes")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrianed weights if not from h5. Could be [imagenet, noisy_student, imagenet21k, imagenet21k-ft1k, imagenet_sam], None for model.pretrained",
    )
    parser.add_argument(
        "--additional_model_kwargs", type=str, default=None, help="Json format model kwargs like '{\"drop_connect_rate\": 0.05}'. Note all quote marks"
    )

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
    input_shape = None if args.input_shape == -1 else (args.input_shape, args.input_shape, 3)

    if args.model_path.startswith("timm."):  # model_path like: timm.models.resmlp_12_224
        import timm

        timm_model_name = ".".join(args.model_path.split(".")[2:])
        model = getattr(timm.models, timm_model_name)(pretrained=True)
    elif args.model_path.endswith(".h5"):
        model = tf.keras.models.load_model(args.model_path, compile=False)
    elif args.model_path.endswith(".tflite"):
        model = args.model_path
    elif args.model_path.endswith(".onnx"):
        model = args.model_path
    else:  # model_path like: volo.VOLO_d1
        model = args.model_path.strip().split(".")
        model_class = getattr(getattr(keras_cv_attention_models, model[0]), model[1])
        model_kwargs = json.loads(args.additional_model_kwargs) if args.additional_model_kwargs else {}
        if input_shape:
            model_kwargs.update({"input_shape": input_shape})
        if args.num_classes:
            model_kwargs.update({"num_classes": args.num_classes})
        if args.pretrained:
            model_kwargs.update({"pretrained": args.pretrained})
        print(">>>> model_kwargs:", model_kwargs)
        model = model_class(**model_kwargs)

    antialias = not args.disable_antialias
    evaluation(model, args.data_name, input_shape, args.batch_size, args.central_crop, args.resize_method, antialias, args.rescale_mode)
