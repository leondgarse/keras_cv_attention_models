#!/usr/bin/env python3
import json
from keras_cv_attention_models.coco.eval_func import run_coco_evaluation
import tensorflow as tf


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Could be: 1. Saved h5 / tflite model path. 2. Model name defined in this repo, format [sub_dir].[model_name] like yolor.YOLOR_CSP",
    )
    parser.add_argument("-i", "--input_shape", type=int, default=-1, help="Model input shape, Set -1 for using model.input_shape")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-d", "--data_name", type=str, default="coco/2017", help="Dataset name from tensorflow_datasets like coco/2017")
    parser.add_argument("--rescale_mode", type=str, default="auto", help="Rescale mode in [tf, torch, raw, raw01]. Default `auto` means using model preset")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    parser.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    parser.add_argument("--use_bgr_input", action="store_true", help="Use BRG input instead of RGB")
    parser.add_argument("--num_classes", type=int, default=None, help="num_classes if not inited from h5 file. None for model.num_classes")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrianed weights if not from h5. Could be coco or specific h5 file, None for model.pretrained",
    )
    parser.add_argument("-F", "--use_anchor_free_mode", action="store_true", help="Use anchor free mode")
    parser.add_argument("-R", "--use_yolor_anchors_mode", action="store_true", help="Use yolor anchors mode")
    parser.add_argument(
        "--anchor_scale", type=int, default=4, help="Anchor scale, base anchor for a single grid point will multiply with it. For efficientdet anchors only"
    )
    # parser.add_argument(
    #     "--anchor_pyramid_levels_min", type=int, default=3, help="Anchor pyramid levels min, max is calculated from model output shape"
    # )

    """ NMS arguments """
    nms_group = parser.add_argument_group("NMS arguments")
    nms_group.add_argument("--nms_score_threshold", type=float, default=0.001, help="nms score threshold")
    nms_group.add_argument("--nms_iou_or_sigma", type=float, default=0.5, help='means `soft_nms_sigma` if nms_method is "gaussian", else `iou_threshold`')
    nms_group.add_argument("--nms_method", type=str, default="gaussian", help="one of [hard, gaussian]")
    nms_group.add_argument(
        "--nms_mode", type=str, default="per_class", help="one of [global, per_class]. `per_class` is strategy from `torchvision.ops.batched_nms`"
    )
    nms_group.add_argument("--nms_topk", type=int, default=5000, help="Using topk highest scores, set `-1` to disable")

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
    antialias = not args.disable_antialias
    data_name = "coco/2017" if args.data_name == "coco" else args.data_name
    ANCHORS = {"anchor_scale": args.anchor_scale, "use_anchor_free_mode": args.use_anchor_free_mode, "use_yolor_anchors_mode": args.use_yolor_anchors_mode}

    if args.model_path.endswith(".h5"):
        model = tf.keras.models.load_model(args.model_path, compile=False)
    elif args.model_path.endswith(".tflite"):
        from keras_cv_attention_models.imagenet.eval_func import TFLiteModelInterf

        # model = args.model_path
        model = TFLiteModelInterf(args.model_path)
        model.output_shape = model(tf.ones([1, *model.input_shape[1:-1], 3])).shape  # Have to init output_shape after inference once, or will be 1 [ ??? ]
    else:  # model_path like: yolor.YOLOR_CSP
        model = args.model_path.strip().split(".")
        model_class = getattr(getattr(keras_cv_attention_models, model[0]), model[1])
        model_kwargs = ANCHORS.copy()
        if input_shape:
            model_kwargs.update({"input_shape": input_shape})
        if args.num_classes:
            model_kwargs.update({"num_classes": args.num_classes})
        if args.pretrained:
            model_kwargs.update({"pretrained": args.pretrained})
        print(">>>> model_kwargs:", model_kwargs)
        model = model_class(**model_kwargs)

    NMS = {"nms_score_threshold": args.nms_score_threshold, "nms_iou_or_sigma": args.nms_iou_or_sigma}
    NMS.update({"nms_method": args.nms_method, "nms_mode": args.nms_mode, "nms_topk": args.nms_topk})
    IMAGE = {"resize_method": args.resize_method, "resize_antialias": antialias, "rescale_mode": args.rescale_mode, "use_bgr_input": args.use_bgr_input}
    print(">>>> COCO evaluation:", data_name, "\n   - image:", IMAGE, "\n   - anchors:", ANCHORS, "\n   - nms:", NMS)
    run_coco_evaluation(model, data_name, input_shape, args.batch_size, **IMAGE, **ANCHORS, **NMS)
