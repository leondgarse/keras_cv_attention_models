#!/usr/bin/env python3
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
    parser.add_argument("--rescale_mode", type=str, default="auto", help="Rescale mode, one of [tf, torch]. Default `auto` means using model preset")
    parser.add_argument("--central_crop", type=float, default=0.95, help="Central crop fraction. Set 1 to disable")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    parser.add_argument("--antialias", action="store_true", help="Set use antialias=True for tf.image.resize")
    parser.add_argument("--num_classes", type=int, default=1000, help="num_classes if not imagenet2012 dataset and not inited from h5 file")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="imagenet",
        help="Pretrianed weights, Other values could be [noisy_student, imagenet21k, imagenet21k-ft1k, imagenet_sam]",
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

        model = getattr(timm.models, args.model_path)(pretrained=True)
    elif args.model_path.endswith(".h5"):
        model = tf.keras.models.load_model(args.model_path, compile=False)
    elif args.model_path.endswith(".tflite"):
        model = args.model_path
    else:  # model_path like: volo.VOLO_d1
        model = args.model_path.strip().split(".")
        model_class = getattr(getattr(keras_cv_attention_models, model[0]), model[1])
        if input_shape:
            model = model_class(num_classes=args.num_classes, input_shape=input_shape, pretrained=args.pretrained)
        else:
            model = model_class(num_classes=args.num_classes, pretrained=args.pretrained)
    evaluation(model, args.data_name, input_shape, args.batch_size, args.central_crop, args.resize_method, args.antialias, args.rescale_mode)
