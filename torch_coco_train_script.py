#!/usr/bin/env python3
import os, sys

os.environ["KECAM_BACKEND"] = "torch"

import torch
import kecam
import pycocotools, cv2, torchvision, tqdm, h5py  # Not using here, just in case for later error


BUILDIN_DATASETS = {
    "coco_dog_cat": {
        "url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/coco_dog_cat.tar.gz",
        "dataset_file": "detections.json",
    },
}
global_device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")


def build_optimizer(model, name="sgd", lr=0.01, momentum=0.937, weight_decay=5e-4):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    name_lower = name.lower()
    if name_lower == "sgd":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name_lower == "adamw":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_name", type=str, default="coco_dog_cat", help="Dataset json file like coco.json")
    parser.add_argument("-i", "--input_shape", type=int, default=640, help="Model input shape")
    parser.add_argument(
        "-B", "--backbone", type=str, default=None, help="Detector backbone, name in format [sub_dir].[model_name]. Default None for header preset."
    )
    parser.add_argument(
        "--backbone_pretrained",
        type=str,
        default="imagenet",
        help="If build backbone with pretrained weights. Mostly one of [imagenet, imagenet21k, noisy_student]",
    )
    parser.add_argument("-D", "--det_header", type=str, default="yolov8.YOLOV8_N", help="Detector header, name in format [sub_dir].[model_name]")
    parser.add_argument(
        "--additional_backbone_kwargs", type=str, default=None, help="Json format backbone kwargs like '{\"drop_connect_rate\": 0.05}'. Note all quote marks"
    )
    parser.add_argument(
        "--additional_det_header_kwargs", type=str, default=None, help="Json format backbone kwargs like '{\"fpn_depth\": 3}'. Note all quote marks"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("-p", "--optimizer", type=str, default="SGD", help="Optimizer name. One of [Adam, SGD].")
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
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Mostly used is `coco`")
    # parser.add_argument("--seed", type=int, default=None, help="Set random seed if not None")
    parser.add_argument("--summary", action="store_true", help="show model summary")
    parser.add_argument("--eval_start_epoch", type=int, default=-1, help="eval process start epoch, default -1 for `epochs * 1 // 4`")

    """ Optimizer arguments like Learning rate, weight decay and momentum """
    lr_group = parser.add_argument_group("Optimizer arguments like Learning rate, weight decay and momentum")
    lr_group.add_argument("--lr_base_512", type=float, default=0.08, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    lr_group.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    lr_group.add_argument("--lr_warmup_steps", type=int, default=3, help="Learning rate warmup epochs")
    lr_group.add_argument("--momentum", type=float, default=0.937, help="Momentum for SGD / SGDW / RMSprop optimizer")

    """ Dataset parameters """
    ds_group = parser.add_argument_group("Dataset arguments")
    ds_group.add_argument("--close_mosaic_epochs", type=int, default=10, help="Epochs closing mosaic mixing in the end of training")
    ds_group.add_argument("--rescale_mode", type=str, default="raw01", help="Rescale mode, one of [tf, torch, raw, raw01]")

    args = parser.parse_known_args(argv)[0]

    args.additional_det_header_kwargs = json.loads(args.additional_det_header_kwargs) if args.additional_det_header_kwargs else {}
    args.additional_backbone_kwargs = json.loads(args.additional_backbone_kwargs) if args.additional_backbone_kwargs else {}

    if args.basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
        args.basic_save_name = basic_save_name
    return args


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    print(">>>> All args:", args)

    """ Dataset """
    if args.data_name in BUILDIN_DATASETS:
        from keras_cv_attention_models.download_and_load import download_buildin_dataset

        args.data_name = download_buildin_dataset(args.data_name, BUILDIN_DATASETS, cache_subdir="datasets")

    train_dataset, _, total_images, num_classes = kecam.coco.data.init_dataset(
        data_path=args.data_name, batch_size=args.batch_size, image_size=args.input_shape, rescale_mode=args.rescale_mode, with_info=True
    )
    image, labels = next(iter(train_dataset))
    print(">>>> total_images: {}, num_classes: {}".format(total_images, num_classes))

    """ Model """
    input_shape = (args.input_shape, args.input_shape, 3)
    if args.backbone is not None:
        backbone = kecam.imagenet.train_func.init_model(args.backbone, input_shape, 0, args.backbone_pretrained, **args.additional_backbone_kwargs)
        args.additional_det_header_kwargs.update({"backbone": backbone})
    args.additional_det_header_kwargs.update({"classifier_activation": None})
    model = kecam.imagenet.train_func.init_model(args.det_header, input_shape, num_classes, args.pretrained, **args.additional_det_header_kwargs)
    if args.summary:
        model.summary()
    model.to(global_device)
    basic_save_name = args.basic_save_name or "{}_{}".format(model.name, os.path.basename(args.data_name))
    print(">>>> basic_save_name:", basic_save_name)
    ema = kecam.imagenet.callbacks.ModelEMA(basic_save_name=basic_save_name, updates=args.initial_epoch * total_images // max(64, args.batch_size))
    ema.set_model(model)

    """ Optimizer, loss and Metrics """
    lr = args.lr_base_512 * args.batch_size / 512
    print(">>>> lr:", lr)
    optimizer = build_optimizer(model, name=args.optimizer, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss = kecam.coco.torch_losses.Loss(device=global_device, nc=num_classes)
    box_loss_metric = kecam.imagenet.metrics.LossMeanMetricWrapper(loss, loss_attr_name="box_loss")
    cls_loss_metric = kecam.imagenet.metrics.LossMeanMetricWrapper(loss, loss_attr_name="cls_loss")
    dfl_loss_metric = kecam.imagenet.metrics.LossMeanMetricWrapper(loss, loss_attr_name="dfl_loss")

    """ Compile """
    if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
        print(">>>> Calling torch.compile")
        model = torch.compile(model)
    grad_accumulate = max(round(64 / args.batch_size), 1)
    metrics = [box_loss_metric, cls_loss_metric, dfl_loss_metric]
    model.train_compile(optimizer=optimizer, loss=loss, metrics=metrics, grad_accumulate=grad_accumulate, grad_max_norm=10.0)

    if args.restore_path is not None:
        print(">>>> Reload weights from:", args.restore_path)
        model.load(args.restore_path)  # Reload wights after compile
        if os.path.exists(ema.save_file_path):
            print(">>>> Reload EMA model weights from:", ema.save_file_path)
            ema.ema.load(ema.save_file_path)

    """ Callback """
    warmup_train = kecam.imagenet.callbacks.WarmupTrain(steps_per_epoch=len(train_dataset), warmup_epochs=args.lr_warmup_steps)
    close_mosaic = kecam.imagenet.callbacks.CloseMosaic(train_dataset, close_mosaic_epoch=args.epochs - args.close_mosaic_epochs)
    start_epoch = args.epochs * 1 // 4 if args.eval_start_epoch < 0 else args.eval_start_epoch  # coco eval starts from 1/4 epochs
    nms_kwargs = {"nms_method": "hard", "nms_iou_or_sigma": 0.65, "nms_max_output_size": 300}
    coco_ap_eval = kecam.coco.eval_func.COCOEvalCallback(args.data_name, args.batch_size, start_epoch=start_epoch, rescale_mode=args.rescale_mode, **nms_kwargs)
    coco_ap_eval.model = ema.ema

    """ Learning rate scheduler and training """
    learning_rate_scheduler = lambda epoch: lr * ((1 - epoch / args.epochs) * (1.0 - lr) + lr)  # linear
    lr_scheduler = kecam.imagenet.callbacks.LearningRateScheduler(learning_rate_scheduler)
    other_kwargs = {}
    latest_save, hist = kecam.imagenet.train_func.train(
        compiled_model=model,
        epochs=args.epochs,
        train_dataset=train_dataset,
        test_dataset=None,
        initial_epoch=args.initial_epoch,
        lr_scheduler=lr_scheduler,
        basic_save_name=basic_save_name,
        init_callbacks=[warmup_train, close_mosaic, ema, coco_ap_eval, ema],
        logs=None,
        **other_kwargs,
    )
