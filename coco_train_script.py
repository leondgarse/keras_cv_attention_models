#!/usr/bin/env python3
import os
import json
from keras_cv_attention_models.coco import data, losses, anchors_func, eval_func
from keras_cv_attention_models.imagenet import (
    compile_model,
    init_global_strategy,
    init_lr_scheduler,
    init_model,
    train,
)
import pycocotools  # Try import first, not using here, just in case it throws error later


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_name", type=str, default="coco/2017", help="Dataset name from tensorflow_datasets like coco/2017")
    parser.add_argument("-i", "--input_shape", type=int, default=256, help="Model input shape")
    parser.add_argument(
        "-B", "--backbone", type=str, default=None, help="Detector backbone, name in format [sub_dir].[model_name]. Default None for header preset."
    )
    parser.add_argument(
        "--backbone_pretrained",
        type=str,
        default="imagenet",
        help="If build backbone with pretrained weights. Mostly one of [imagenet, imagenet21k, noisy_student]",
    )
    parser.add_argument("-D", "--det_header", type=str, default="efficientdet.EfficientDetD0", help="Detector header, name in format [sub_dir].[model_name]")
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
    parser.add_argument(
        "-s",
        "--basic_save_name",
        type=str,
        default=None,
        help="Basic save name for model and history. None means a combination of parameters, or starts with _ as a suffix to default name",
    )
    parser.add_argument(
        "-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 by `keras.models.load_model` directly. Higher priority than model"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="""If build model with pretrained weights. Mostly used is `coco`. Or specified h5 file for build model -> restore weights.
                This will drop model optimizer, used for `progressive_train_script.py`. Relatively, `restore_path` is used for restore from break point""",
    )
    parser.add_argument("--seed", type=int, default=None, help="Set random seed if not None")
    parser.add_argument("--summary", action="store_true", help="show model summary")
    parser.add_argument("--disable_float16", action="store_true", help="Disable mixed_float16 training")
    parser.add_argument("--TPU", action="store_true", help="Run training on TPU [Not working]")
    parser.add_argument(
        "--tensorboard_logs",
        type=str,
        default=None,
        help="TensorBoard logs saving path, default None for disable. Set auto for `logs/{basic_save_name} + _ + timestamp`.",
    )
    parser.add_argument("--eval_start_epoch", type=int, default=-1, help="eval process start epoch, default -1 for `epochs * 2 // 3`")

    """ Anchor arguments """
    anchor_group = parser.add_argument_group("Anchor arguments")
    anchor_group.add_argument(
        "-A", "--anchors_mode", type=str, default=None, help="One of [efficientdet, anchor_free, yolor]. Default None for det_header preset"
    )
    anchor_group.add_argument(
        "--anchor_scale", type=int, default=4, help="For efficientdet anchors only. base anchor for a single grid point will multiply with it."
    )
    anchor_group.add_argument("--aspect_ratios", type=float, nargs="*", default=(1, 2, 0.5), help="For efficientdet anchors only. anchors aspect ratio")
    anchor_group.add_argument("--num_scales", type=int, default=3, help="For efficientdet anchors only. number of scale for each aspect_ratios")
    anchor_group.add_argument("--anchor_pyramid_levels_min", type=int, default=3, help="Anchor pyramid levels min.")
    anchor_group.add_argument("--anchor_pyramid_levels_max", type=int, default=-1, help="Anchor pyramid levels max. -1 for calculated from model output shape")

    """ Loss arguments """
    loss_group = parser.add_argument_group("Loss arguments")
    loss_group.add_argument("--label_smoothing", type=float, default=0, help="Loss label smoothing value")
    loss_group.add_argument("--use_l1_loss", action="store_true", help="Use additional l1_loss. For anchor_free mode only")
    loss_group.add_argument("--bbox_loss_weight", type=float, default=-1, help="Bbox loss weight, -1 means using loss preset values.")

    """ Optimizer arguments like Learning rate, weight decay and momentum """
    lr_group = parser.add_argument_group("Optimizer arguments like Learning rate, weight decay and momentum")
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
    lr_group.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD / SGDW / RMSprop optimizer")

    """ Dataset parameters """
    ds_group = parser.add_argument_group("Dataset arguments")
    ds_group.add_argument("--magnitude", type=int, default=6, help="Positional Randaug magnitude value, including rotate / shear / transpose")
    ds_group.add_argument("--num_layers", type=int, default=2, help="Number of randaug applied sequentially to an image. Usually best in [1, 3]")
    ds_group.add_argument(
        "--color_augment_method", type=str, default="random_hsv", help="None positional related augment method, one of [random_hsv, autoaug, randaug]"
    )
    ds_group.add_argument(
        "--positional_augment_methods",
        type=str,
        default="rts",
        help="Positional related augment method besides random scale, combine of r: rotate, t: transplate, s: shear, x: scale_x + scale_y",
    )
    ds_group.add_argument("--mosaic_mix_prob", type=float, default=0.5, help="Mosaic mix probability, 0 to disable")
    ds_group.add_argument(
        "--random_crop_mode", type=float, default=0.5, help="0 for eval mode, (0, 1) for random crop, 1 for random largest crop, > 1 for random scale"
    )
    ds_group.add_argument("--rescale_mode", type=str, default="torch", help="Rescale mode, one of [tf, torch, raw, raw01]")
    ds_group.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    ds_group.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    ds_group.add_argument("--max_labels_per_image", type=int, default=100, help="Max number of ground truth labels used in a single image")

    args = parser.parse_known_args(argv)[0]

    args.additional_det_header_kwargs = json.loads(args.additional_det_header_kwargs) if args.additional_det_header_kwargs else {}
    if args.anchors_mode is not None:
        args.additional_det_header_kwargs.update({"anchors_mode": args.anchors_mode})
    args.efficient_det_num_anchors = len(args.aspect_ratios) * args.num_scales
    if args.anchors_mode == anchors_func.EFFICIENTDET_MODE:
        args.additional_det_header_kwargs.update({"num_anchors": args.efficient_det_num_anchors})
    args.additional_backbone_kwargs = json.loads(args.additional_backbone_kwargs) if args.additional_backbone_kwargs else {}

    lr_decay_steps = args.lr_decay_steps.strip().split(",")
    if len(lr_decay_steps) > 1:
        # Constant decay steps
        args.lr_decay_steps = [int(ii.strip()) for ii in lr_decay_steps if len(ii.strip()) > 0]
    else:
        # Cosine decay
        args.lr_decay_steps = int(lr_decay_steps[0].strip())

    if args.basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
        args.basic_save_name = basic_save_name
    elif args.basic_save_name is None or args.basic_save_name.startswith("_"):
        data_name = args.data_name.replace("/", "_")
        model_name = args.det_header.split(".")[-1] + ("" if args.backbone is None else ("_" + args.backbone.split(".")[-1]))
        basic_save_name = "{}_{}_{}_{}_batchsize_{}".format(model_name, args.input_shape, args.optimizer, data_name, args.batch_size)
        basic_save_name += "_randaug_{}_mosaic_{}".format(args.magnitude, args.mosaic_mix_prob)
        basic_save_name += "_color_{}_position_{}".format(args.color_augment_method, args.positional_augment_methods)
        basic_save_name += "_lr512_{}_wd_{}_anchors_mode_{}".format(args.lr_base_512, args.weight_decay, args.anchors_mode)
        args.basic_save_name = basic_save_name if args.basic_save_name is None else (basic_save_name + args.basic_save_name)
    args.enable_float16 = not args.disable_float16
    args.tensorboard_logs = None if args.tensorboard_logs is None or args.tensorboard_logs.lower() == "none" else args.tensorboard_logs

    return args


def run_training_by_args(args):
    print(">>>> All args:", args)

    strategy = init_global_strategy(args.enable_float16, args.seed, args.TPU)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (args.input_shape, args.input_shape, 3)

    # Init model first, for getting actual pyramid_levels
    total_images, num_classes, steps_per_epoch = data.init_dataset(args.data_name, batch_size=batch_size, info_only=True)
    with strategy.scope():
        if args.backbone is not None:
            backbone = init_model(args.backbone, input_shape, 0, args.backbone_pretrained, **args.additional_backbone_kwargs)
            args.additional_det_header_kwargs.update({"backbone": backbone})
        det_header = args.det_header if args.restore_path is None else args.restore_path
        model = init_model(det_header, input_shape, num_classes, args.pretrained, **args.additional_det_header_kwargs)
        if args.summary:
            model.summary()

    total_anchors = model.output_shape[1]
    if args.anchors_mode is None or args.anchors_mode == "auto":
        args.anchors_mode, num_anchors = anchors_func.get_anchors_mode_by_anchors(input_shape, total_anchors=total_anchors)
    elif args.anchors_mode == anchors_func.EFFICIENTDET_MODE:
        num_anchors = args.efficient_det_num_anchors
    else:
        num_anchors = anchors_func.NUM_ANCHORS.get(args.anchors_mode, 9)

    if args.anchor_pyramid_levels_max <= 0:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(input_shape, total_anchors, num_anchors, args.anchor_pyramid_levels_min)
        args.anchor_pyramid_levels_max = max(pyramid_levels)
    args.anchor_pyramid_levels = [args.anchor_pyramid_levels_min, args.anchor_pyramid_levels_max]
    print(">>>> anchor_pyramid_levels: {}, anchors_mode: {}, num_anchors: {}".format(args.anchor_pyramid_levels, args.anchors_mode, num_anchors))

    resize_antialias = not args.disable_antialias
    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        max_labels_per_image=args.max_labels_per_image,
        anchors_mode=args.anchors_mode,
        anchor_pyramid_levels=args.anchor_pyramid_levels,
        anchor_scale=args.anchor_scale,
        aspect_ratios=args.aspect_ratios,
        num_scales=args.num_scales,
        rescale_mode=args.rescale_mode,
        resize_method=args.resize_method,
        resize_antialias=resize_antialias,
        random_crop_mode=args.random_crop_mode,
        mosaic_mix_prob=args.mosaic_mix_prob,
        color_augment_method=args.color_augment_method,  # If would like custom augmentation function, just pass a function like `lambda image: image`
        positional_augment_methods=args.positional_augment_methods,
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
        if model.optimizer is None:
            loss_kwargs = {"label_smoothing": args.label_smoothing}
            if args.bbox_loss_weight > 0:
                loss_kwargs.update({"bbox_loss_weight": args.bbox_loss_weight})

            if args.anchors_mode == anchors_func.ANCHOR_FREE_MODE:  # == "anchor_free"
                loss = losses.AnchorFreeLoss(input_shape, args.anchor_pyramid_levels, use_l1_loss=args.use_l1_loss, **loss_kwargs)
            elif args.anchors_mode == anchors_func.YOLOR_MODE:  # == "yolor"
                loss = losses.YOLORLossWithBbox(input_shape, args.anchor_pyramid_levels, **loss_kwargs)
            else:
                # loss, metrics = losses.FocalLossWithBbox(label_smoothing=args.label_smoothing), losses.ClassAccuracyWithBbox()
                loss = losses.FocalLossWithBbox(**loss_kwargs)
            metrics = losses.ClassAccuracyWithBboxWrapper(loss)
            model = compile_model(model, args.optimizer, lr_base, args.weight_decay, loss=loss, metrics=metrics, momentum=args.momentum)
        else:
            # Re-compile the metrics after restore from h5
            metrics = losses.ClassAccuracyWithBboxWrapper(model.loss)
            model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics, momentum=args.momentum)
        print(">>>> basic_save_name =", args.basic_save_name)
        # return None, None, None

        # Save line width...
        kw = {"batch_size": batch_size, "rescale_mode": args.rescale_mode, "resize_method": args.resize_method, "resize_antialias": resize_antialias}
        kw.update({"anchor_scale": args.anchor_scale, "anchors_mode": args.anchors_mode, "model_basic_save_name": args.basic_save_name})
        kw.update({"aspect_ratios": args.aspect_ratios, "num_scales": args.num_scales, "nms_max_output_size": args.max_labels_per_image})
        start_epoch = epochs * 2 // 3 if args.eval_start_epoch < 0 else args.eval_start_epoch  # coco eval starts from 2/3 epochs
        frequency = 1
        print(">>>> COCO AP eval start_epoch: {}, frequency: {}".format(start_epoch, frequency))
        coco_ap_eval = eval_func.COCOEvalCallback(args.data_name, start_epoch=start_epoch, frequency=frequency, **kw)

        init_callbacks = [coco_ap_eval]
        test_dataset = None  # COCO eval using coco_ap_eval callback, set `validation_data` for `model.fit` to None
        latest_save, hist = train(
            model, epochs, train_dataset, test_dataset, args.initial_epoch, lr_scheduler, args.basic_save_name, init_callbacks, logs=args.tensorboard_logs
        )
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
