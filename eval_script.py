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
    parser.add_argument("--rescale_mode", type=str, default="auto", help="Rescale mode, one of [tf, torch, raw]. Default `auto` means using model preset")
    parser.add_argument("--central_crop", type=float, default=0.95, help="Central crop fraction. Set 1 to disable")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    parser.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    parser.add_argument("--num_classes", type=int, default=None, help="num_classes if not inited from h5 file. None for model.num_classes")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrianed weights if not from h5. Could be [imagenet, noisy_student, imagenet21k, imagenet21k-ft1k, imagenet_sam], None for model.pretrained",
    )

    """ Anchor arguments """
    anchor_group = parser.add_argument_group("COCO arguments")
    anchor_group.add_argument("--use_anchor_free_mode", action="store_true", help="[COCO] Use anchor free mode")
    # anchor_group.add_argument(
    #     "--anchor_pyramid_levels_min", type=int, default=3, help="[COCO] Anchor pyramid levels min, max is calculated from model output shape"
    # )
    anchor_group.add_argument(
        "--anchor_scale", type=int, default=4, help="Anchor scale, base anchor for a single grid point will multiply with it. Force 1 if use_anchor_free_mode"
    )
    anchor_group.add_argument(
        "--additional_anchor_kwargs", type=str, default=None, help="Json format anchor kwargs like '{\"nms_method\": \"hard\"}'. Note all quote marks"
    )

    args = parser.parse_known_args(argv)[0]

    args.additional_anchor_kwargs = json.loads(args.additional_anchor_kwargs) if args.additional_anchor_kwargs else {}
    if args.use_anchor_free_mode:
        args.anchor_scale = 1
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
    else:  # model_path like: volo.VOLO_d1
        model = args.model_path.strip().split(".")
        model_class = getattr(getattr(keras_cv_attention_models, model[0]), model[1])
        model_kwargs = {}
        if input_shape:
            model_kwargs.update({"input_shape": input_shape})
        if args.num_classes:
            model_kwargs.update({"num_classes": args.num_classes})
        if args.pretrained:
            model_kwargs.update({"pretrained": args.pretrained})
        print(">>>> model_kwargs:", model_kwargs)
        model = model_class(**model_kwargs)

    antialias = not args.disable_antialias
    if args.data_name.startswith("coco"):
        from keras_cv_attention_models.coco.eval_func import run_coco_evaluation

        data_name = "coco/2017" if args.data_name == "coco" else args.data_name
        ANCHORS = {"anchor_scale": args.anchor_scale, "use_anchor_free_mode": args.use_anchor_free_mode}
        print(">>>> COCO evaluation:", data_name, "- anchors:", ANCHORS)
        run_coco_evaluation(
            model, data_name, input_shape, args.batch_size, args.resize_method, antialias, args.rescale_mode, **ANCHORS, **args.additional_anchor_kwargs
        )
    else:
        evaluation(model, args.data_name, input_shape, args.batch_size, args.central_crop, args.resize_method, antialias, args.rescale_mode)
