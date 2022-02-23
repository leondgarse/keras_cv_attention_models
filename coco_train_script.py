#!/usr/bin/env python3
import os
import json
from keras_cv_attention_models.coco import data, losses
from keras_cv_attention_models.imagenet import (
    compile_model,
    init_global_strategy,
    init_lr_scheduler,
    init_model,
    train,
)


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_name", type=str, default="coco/2017", help="Dataset name from tensorflow_datasets like coco/2017")
    parser.add_argument("-i", "--input_shape", type=int, default=256, help="Model input shape")
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
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=-1, help="Total epochs. Set -1 means using lr_decay_steps + lr_cooldown_steps")
    parser.add_argument("-p", "--optimizer", type=str, default="LAMB", help="Optimizer name. One of [AdamW, LAMB, RMSprop, SGD, SGDW].")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default=None, help="Basic save name for model and history. None means a combination of parameters")
    parser.add_argument(
        "-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 by `keras.models.load_model` directly. Higher priority than model"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="""If build model with pretrained weights. Mostly used is one of [imagenet, imagenet21k]. Or specified h5 file for build model -> restore weights.
                This will drop model optimizer, used for `progressive_train_script.py`. Relatively, `restore_path` is used for restore from break point""",
    )
    parser.add_argument("--seed", type=int, default=None, help="Set random seed if not None")
    parser.add_argument("--summary", action="store_true", help="show model summary")
    parser.add_argument("--disable_float16", action="store_true", help="Disable mixed_float16 training")
    parser.add_argument("--TPU", action="store_true", help="Run training on TPU [Not working]")

    """ Anchor arguments """
    anchor_group = parser.add_argument_group("Anchor arguments")
    anchor_group.add_argument("--anchor_scale", type=int, default=4, help="Anchor scale, base anchor for a single grid point will multiply with this")
    anchor_group.add_argument(
        "--anchor_num_scales", type=int, default=3, help="Anchor num scales, `scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]`"
    )
    anchor_group.add_argument(
        "--anchor_aspect_ratios",
        type=float,
        nargs="+",
        default=[1, 2, 0.5],
        help="Anchor aspect ratios, `num_anchors = len(anchor_aspect_ratios) * anchor_num_scales`",
    )
    anchor_group.add_argument("--anchor_pyramid_levels", type=int, nargs="+", default=[3, 7], help="Anchor pyramid levels, 2 values indicates min max")

    """ Loss arguments """
    loss_group = parser.add_argument_group("Loss arguments")
    loss_group.add_argument("--label_smoothing", type=float, default=0, help="Loss label smoothing value")

    """ Learning rate and weight decay arguments """
    lr_group = parser.add_argument_group("Learning rate and weight decay arguments")
    lr_group.add_argument("--lr_base_512", type=float, default=8e-3, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    lr_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.02,
        help="Weight decay. For SGD, it's L2 value. For AdamW / SGDW, it will multiply with learning_rate. For LAMB, it's directly used",
    )
    lr_group.add_argument(
        "--lr_decay_steps",
        type=str,
        default="100",
        help="Learning rate decay epoch steps. Single value like 100 for cosine decay. Set 30,60,90 for constant decay steps",
    )
    lr_group.add_argument("--lr_decay_on_batch", action="store_true", help="Learning rate decay on each batch, or on epoch")
    lr_group.add_argument("--lr_warmup", type=float, default=1e-4, help="Learning rate warmup value")
    lr_group.add_argument("--lr_warmup_steps", type=int, default=5, help="Learning rate warmup epochs")
    lr_group.add_argument("--lr_cooldown_steps", type=int, default=5, help="Learning rate cooldown epochs")
    lr_group.add_argument("--lr_min", type=float, default=1e-6, help="Learning rate minimum value")
    lr_group.add_argument("--lr_t_mul", type=float, default=2, help="For CosineDecayRestarts, derive the number of iterations in the i-th period")
    lr_group.add_argument("--lr_m_mul", type=float, default=0.5, help="For CosineDecayRestarts, derive the initial learning rate of the i-th period")

    """ Dataset parameters """
    ds_group = parser.add_argument_group("Dataset arguments")
    ds_group.add_argument("--magnitude", type=int, default=6, help="Randaug magnitude value")
    ds_group.add_argument("--num_layers", type=int, default=2, help="Number of randaug applied sequentially to an image. Usually best in [1, 3]")
    ds_group.add_argument("--random_crop_min", type=float, default=0.08, help="Random crop min value for RRC. Set 1 to disable RRC")
    ds_group.add_argument("--rescale_mode", type=str, default="torch", help="Rescale mode, one of [tf, torch]")
    ds_group.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    ds_group.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")

    args = parser.parse_known_args(argv)[0]

    args.additional_det_header_kwargs = json.loads(args.additional_det_header_kwargs) if args.additional_det_header_kwargs else {}
    args.additional_det_header_kwargs.update(
        {
            "pyramid_levels": args.anchor_pyramid_levels,
            "anchor_scale": args.anchor_scale,
            "num_anchors": len(args.anchor_aspect_ratios) * args.anchor_num_scales,
        }
    )
    args.additional_backbone_kwargs = json.loads(args.additional_backbone_kwargs) if args.additional_backbone_kwargs else {}

    lr_decay_steps = args.lr_decay_steps.strip().split(",")
    if len(lr_decay_steps) > 1:
        # Constant decay steps
        args.lr_decay_steps = [int(ii.strip()) for ii in lr_decay_steps if len(ii.strip()) > 0]
    else:
        # Cosine decay
        args.lr_decay_steps = int(lr_decay_steps[0].strip())

    basic_save_name = args.basic_save_name
    if basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
    elif basic_save_name is None:
        data_name = args.data_name.replace("/", "_")
        model_name = args.det_header.split(".")[-1] + "_" + args.backbone.split(".")[-1]
        basic_save_name = "{}_{}_{}_{}_batchsize_{}".format(model_name, args.input_shape, args.optimizer, data_name, args.batch_size)
        basic_save_name += "_randaug_{}_RRC_{}".format(args.magnitude, args.random_crop_min)
        basic_save_name += "_lr512_{}_wd_{}".format(args.lr_base_512, args.weight_decay)
    args.basic_save_name = basic_save_name
    args.enable_float16 = not args.disable_float16

    return args


def run_training_by_args(args):
    print(">>>> All args:", args)

    strategy = init_global_strategy(args.enable_float16, args.seed, args.TPU)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (args.input_shape, args.input_shape, 3)

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        anchor_pyramid_levels=args.anchor_pyramid_levels,
        anchor_aspect_ratios=args.anchor_aspect_ratios,
        anchor_num_scales=args.anchor_num_scales,
        anchor_scale=args.anchor_scale,
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
        pretrained, restore_path = args.pretrained, args.restore_path
        backbone = init_model(args.backbone, input_shape, 0, args.backbone_pretrained, None, **args.additional_backbone_kwargs)
        args.additional_det_header_kwargs.update({"backbone": backbone})
        model = init_model(args.det_header, None, num_classes, pretrained, restore_path, **args.additional_det_header_kwargs)

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

    args = parse_arguments(sys.argv[1:])
    cyan_print = lambda ss: print("\033[1;36m" + ss + "\033[0m")

    if args.freeze_backbone_epochs - args.initial_epoch > 0:
        total_epochs = args.epochs
        cyan_print(">>>> Train with freezing backbone")
        args.additional_det_header_kwargs.update({"freeze_backbone": True})
        args.epochs = args.freeze_backbone_epochs
        model, latest_save, _ = run_training_by_args(args)

        cyan_print(">>>> Unfreezing backbone")
        args.additional_det_header_kwargs.update({"freeze_backbone": False})
        args.initial_epoch = args.freeze_backbone_epochs
        args.epochs = total_epochs
        args.backbone_pretrained = None
        args.restore_path = None
        args.pretrained = latest_save  # Build model and load weights

    run_training_by_args(args)
