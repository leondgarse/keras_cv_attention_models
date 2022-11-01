#!/usr/bin/env python3
import json
from keras_cv_attention_models.coco import anchors_func, eval_func
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
    parser.add_argument("-s", "--save_json", type=str, default=None, help="Save detection results to json file, None for not saving")
    parser.add_argument("--rescale_mode", type=str, default="auto", help="Rescale mode in [tf, torch, raw, raw01]. Default `auto` means using model preset")
    parser.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    parser.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    parser.add_argument(
        "--letterbox_pad",
        type=int,
        default=-1,
        help="Wrapper resized image in the center."
        + " For input_shape=640, letterbox_pad=0, image shape=(480, 240), will first resize to (640, 320), then pad top=0 left=160 bottom=0 right=160."
        + " For input_shape=704, letterbox_pad=64, image shape=(480, 240), will first resize to (640, 320), then pad top=32 left=192 bottom=32 right=192."
        + " Default -1 for disable",
    )
    parser.add_argument("--use_bgr_input", action="store_true", help="Use BRG input instead of RGB")
    parser.add_argument("--num_classes", type=int, default=None, help="num_classes if not inited from h5 file. None for model.num_classes")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrianed weights if not from h5. Could be coco or specific h5 file, None for model.pretrained",
    )
    parser.add_argument(
        "-A", "--anchors_mode", type=str, default=None, help="One of [efficientdet, anchor_free, yolor]. Default None for calculated from model output_shape"
    )
    parser.add_argument(
        "--anchor_scale", type=int, default=4, help="Anchor scale, base anchor for a single grid point will multiply with it. For efficientdet anchors only"
    )
    parser.add_argument("--aspect_ratios", type=float, nargs="*", default=(1, 2, 0.5), help="For efficientdet anchors only. anchors aspect ratio")
    parser.add_argument("--num_scales", type=int, default=3, help="For efficientdet anchors only. number of scale for each aspect_ratios")
    # parser.add_argument(
    #     "--anchor_pyramid_levels_min", type=int, default=3, help="Anchor pyramid levels min, max is calculated from model output shape"
    # )

    """ NMS arguments """
    nms_group = parser.add_argument_group("NMS arguments")
    nms_group.add_argument("--nms_score_threshold", type=float, default=0.001, help="nms score threshold")
    nms_group.add_argument("--nms_iou_or_sigma", type=float, default=0.5, help='means `soft_nms_sigma` if nms_method is "gaussian", else `iou_threshold`')
    nms_group.add_argument("--nms_max_output_size", type=int, default=100, help="max_output_size for tf.image.non_max_suppression_with_scores")
    nms_group.add_argument("--nms_method", type=str, default="gaussian", help="one of [hard, gaussian]")
    nms_group.add_argument(
        "--nms_mode", type=str, default="per_class", help="one of [global, per_class]. `per_class` is strategy from `torchvision.ops.batched_nms`"
    )
    nms_group.add_argument(
        "--nms_topk", type=int, default=5000, help="Using topk highest scores, each bbox may have multi labels. Set `0` to disable, `-1` using all"
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

    if args.model_path.endswith(".json"):
        # Reloaded from previous saved json results
        with open(args.model_path, "r") as ff:
            detection_results = json.load(ff)
        eval_func.coco_evaluation(detection_results)
        sys.exit(0)

    input_shape = None if args.input_shape == -1 else (args.input_shape, args.input_shape, 3)
    antialias = not args.disable_antialias
    data_name = "coco/2017" if args.data_name == "coco" else args.data_name

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
        model_kwargs = {"anchor_scale": args.anchor_scale}
        if args.anchors_mode is not None and args.anchors_mode != "auto":
            model_kwargs.update({"anchors_mode": args.anchors_mode})
        if args.anchors_mode == anchors_func.EFFICIENTDET_MODE:
            model_kwargs.update({"num_anchors": len(args.aspect_ratios) * args.num_scales})
        if input_shape:
            model_kwargs.update({"input_shape": input_shape})
        if args.num_classes:
            model_kwargs.update({"num_classes": args.num_classes})
        if args.pretrained:
            model_kwargs.update({"pretrained": args.pretrained})
        print(">>>> model_kwargs:", model_kwargs)
        model = model_class(**model_kwargs)

    ANCHORS = {"anchor_scale": args.anchor_scale, "anchors_mode": args.anchors_mode, "aspect_ratios": args.aspect_ratios, "num_scales": args.num_scales}
    NMS = {"nms_score_threshold": args.nms_score_threshold, "nms_iou_or_sigma": args.nms_iou_or_sigma, "nms_max_output_size": args.nms_max_output_size}
    NMS.update({"nms_method": args.nms_method, "nms_mode": args.nms_mode, "nms_topk": args.nms_topk})
    IMAGE = {"resize_method": args.resize_method, "resize_antialias": antialias, "rescale_mode": args.rescale_mode, "use_bgr_input": args.use_bgr_input}
    print(">>>> COCO evaluation:", data_name, "\n   - image:", IMAGE, "\n   - anchors:", ANCHORS, "\n   - nms:", NMS)

    ee = eval_func.COCOEvalCallback(data_name, args.batch_size, **IMAGE, **ANCHORS, **NMS, letterbox_pad=args.letterbox_pad, save_json=args.save_json)
    ee.model = model
    ee.on_epoch_end()
