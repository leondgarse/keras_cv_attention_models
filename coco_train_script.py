#!/usr/bin/env python3
import os
import json
import tensorflow as tf
from train_script import parse_arguments
from keras_cv_attention_models.coco import data, losses
from keras_cv_attention_models.imagenet import (
    compile_model,
    init_global_strategy,
    init_lr_scheduler,
    init_model,
    train,
)


def coco_train_parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_shape", type=int, default=256, help="Model input shape")
    parser.add_argument("-d", "--data_name", type=str, default="coco/2017", help="Dataset name from tensorflow_datasets like coco/2017")
    parser.add_argument("--backbone", type=str, default="efficientnet.EfficientNetV1B0", help="Detector backbone model, name in format [sub_dir].[model_name]")
    parser.add_argument(
        "--backbone_pretrained",
        type=str,
        default="imagenet",
        help="If build backbone with pretrained weights. Mostly one of [imagenet, imagenet21k, noisy_student]",
    )
    parser.add_argument("--det_header", type=str, default="efficientdet.EfficientDet", help="Detector header, name in format [sub_dir].[model_name]")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=32, help="Epochs training with backbone.trainable=false")
    parser.add_argument(
        "--additional_backbone_kwargs", type=str, default=None, help="Json format backbone kwargs like '{\"drop_connect_rate\": 0.05}'. Note all quote marks"
    )
    parser.add_argument(
        "--additional_det_header_kwargs", type=str, default=None, help="Json format backbone kwargs like '{\"fpn_depth\": 3}'. Note all quote marks"
    )

    anchor_group = parser.add_argument_group("Anchor arguments")
    anchor_group.add_argument("--anchor_scale", type=int, default=4, help="Anchor scale, base anchor for a single grid point will multiply with this")
    anchor_group.add_argument(
        "--anchor_num_scales", type=int, default=3, help="Anchor num scales, `scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]`"
    )
    anchor_group.add_argument(
        "--anchor_aspect_ratios",
        type=float,
        nargs="+",
        default=[0.5, 1, 2],
        help="Anchor aspect ratios, `num_anchors = len(anchor_aspect_ratios) * anchor_num_scales`",
    )
    anchor_group.add_argument("--anchor_pyramid_levels", type=int, nargs="+", default=[3, 7], help="Anchor pyramid levels, 2 values indicates min max")

    if "-h" in argv or "--help" in argv:
        parser.print_help()
        print("")
        print(">>>> train_script.py arguments:")
        parse_arguments(argv, is_from_coco=True)
    coco_args, train_argv = parser.parse_known_args(argv)

    model_name = coco_args.det_header.split(".")[-1] + "_" + coco_args.backbone.split(".")[-1]
    train_argv += ["--COCO_NAME", "{}_{}_{}".format(model_name, coco_args.input_shape, coco_args.data_name).replace("/", "_")]
    # print(train_argv)
    train_args = parse_arguments(train_argv, is_from_coco=True)

    coco_args.additional_det_header_kwargs = json.loads(coco_args.additional_det_header_kwargs) if coco_args.additional_det_header_kwargs else {}
    coco_args.additional_det_header_kwargs.update(
        {
            "pyramid_levels": coco_args.anchor_pyramid_levels,
            "anchor_scale": coco_args.anchor_scale,
            "num_anchors": len(coco_args.anchor_aspect_ratios) * coco_args.anchor_num_scales,
        }
    )
    coco_args.additional_backbone_kwargs = json.loads(coco_args.additional_backbone_kwargs) if coco_args.additional_backbone_kwargs else {}

    return coco_args, train_args


def run_training_by_args(coco_args, args):
    print(">>>> COCO args:", coco_args)
    print(">>>> Train args:", args)

    strategy = init_global_strategy(args.enable_float16, args.seed, args.TPU)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (coco_args.input_shape, coco_args.input_shape, 3)

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=coco_args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        anchor_pyramid_levels=coco_args.anchor_pyramid_levels,
        anchor_aspect_ratios=coco_args.anchor_aspect_ratios,
        anchor_num_scales=coco_args.anchor_num_scales,
        anchor_scale=coco_args.anchor_scale,
        rescale_mode=args.rescale_mode,
        random_crop_min=args.random_crop_min,
        resize_method=args.resize_method,
        resize_antialias=not args.disable_antialias,
        magnitude=args.magnitude,
        num_layers=args.num_layers,
    )

    lr_base = args.lr_base_512 * batch_size / 512
    warmup_steps, cooldown_steps, t_mul, m_mul = args.lr_warmup_steps, args.lr_cooldown_steps, args.lr_t_mul, args.lr_m_mul  # Save line-width
    lr_scheduler, lr_total_epochs = init_lr_scheduler(
        lr_base, args.lr_decay_steps, args.lr_min, args.lr_decay_on_batch, args.lr_warmup, warmup_steps, cooldown_steps, t_mul, m_mul
    )
    epochs = args.epochs if args.epochs != -1 else lr_total_epochs

    with strategy.scope():
        pretrained, restore_path = train_args.pretrained, train_args.restore_path
        backbone = init_model(coco_args.backbone, input_shape, 0, coco_args.backbone_pretrained, None, **coco_args.additional_backbone_kwargs)
        coco_args.additional_det_header_kwargs.update({"backbone": backbone})
        model = init_model(coco_args.det_header, None, num_classes, pretrained, restore_path, **coco_args.additional_det_header_kwargs)

        if args.summary:
            model.summary()
        if model.optimizer is None:
            loss, metrics = losses.FocalLossWithBbox(), losses.ClassAccuracyWithBbox()
            model = compile_model(model, args.optimizer, lr_base, args.weight_decay, 1, args.label_smoothing, loss=loss, metrics=metrics)
        print(">>>> basic_save_name =", args.basic_save_name)
        # return None, None, None
        latest_save, hist = train(model, epochs, train_dataset, test_dataset, args.initial_epoch, lr_scheduler, args.basic_save_name)
    return model, latest_save, hist


if __name__ == "__main__":
    import sys

    coco_args, train_args = coco_train_parse_arguments(sys.argv[1:])
    cyan_print = lambda ss: print("\033[1;36m" + ss + "\033[0m")

    if coco_args.freeze_backbone_epochs - train_args.initial_epoch > 0:
        total_epochs = train_args.epochs
        cyan_print(">>>> Train with freezing backbone")
        coco_args.additional_det_header_kwargs.update({"freeze_backbone": True})
        train_args.epochs = coco_args.freeze_backbone_epochs
        model, latest_save, _ = run_training_by_args(coco_args, train_args)

        cyan_print(">>>> Unfreezing backbone")
        coco_args.additional_det_header_kwargs.update({"freeze_backbone": False})
        train_args.initial_epoch = coco_args.freeze_backbone_epochs
        train_args.epochs = total_epochs
        coco_args.backbone_pretrained = None
        train_args.restore_path = None
        train_args.pretrained = latest_save  # Build model and load weights

    run_training_by_args(coco_args, train_args)
