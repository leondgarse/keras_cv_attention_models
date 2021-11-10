#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.imagenet import init_dataset
from keras_cv_attention_models.imagenet import init_lr_scheduler, init_model, compile_model, train


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_shape", type=int, default=160, help="Model input shape")
    parser.add_argument("-m", "--model", type=str, default="aotnet.AotNet50", help="Model name defined in this repo, format [sub_dir].[model_name]")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=105, help="Total epochs")
    parser.add_argument("-d", "--data_name", type=str, default="imagenet2012", help="Dataset name from tensorflow_datasets like imagenet2012 cifar10")
    parser.add_argument("-p", "--optimizer", type=str, default="LAMB", help="Optimizer name. One of [SGD, LAMB, AdamW].")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default=None, help="Basic save name for model and history. None means a combination of parameters")
    parser.add_argument("-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 file. Higher priority than model")
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights")
    parser.add_argument("--seed", type=int, default=None, help="Set random seed if not None")

    """ Loss arguments """
    loss_group = parser.add_argument_group("Loss arguments")
    loss_group.add_argument("--label_smoothing", type=float, default=0, help="Loss label smoothing value")
    loss_group.add_argument(
        "--bce_threshold", type=float, default=1, help="Value [0, 1) for BCE loss target_threshold, otherwise using CategoricalCrossentropy"
    )

    """ Learning rate and weight decay arguments """
    lr_group = parser.add_argument_group("Learning rate and weight decay arguments")
    lr_group.add_argument("--lr_base_512", type=float, default=8e-3, help="Learning rate for batch_size=512")
    lr_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.02,
        help="Weight decay. For SGD, it's L2 value. For AdamW, it will multiply with learning_rate. For LAMB, it's directly used",
    )
    lr_group.add_argument(
        "--lr_decay_steps",
        type=str,
        default="100",
        help="Learning rate decay epoch steps. Single value like 100 for cosine decay. Set 30,60,90 for constant decay steps",
    )
    lr_group.add_argument("--lr_decay_on_batch", action="store_true", help="Learning rate decay on each batch, or on epoch")
    lr_group.add_argument("--lr_warmup", type=int, default=5, help="Learning rate warmup epochs")
    lr_group.add_argument("--lr_min", type=float, default=1e-6, help="Learning rate minimum value")

    """ Dataset parameters """
    ds_group = parser.add_argument_group("Dataset arguments")
    ds_group.add_argument("--magnitude", type=int, default=6, help="Randaug magnitude value")
    ds_group.add_argument("--num_layers", type=int, default=2, help="Number of randaug applied sequentially to an image. Usually best in [1, 3]")
    ds_group.add_argument("--mixup_alpha", type=float, default=0.1, help="Mixup alpha value")
    ds_group.add_argument("--cutmix_alpha", type=float, default=1.0, help="Cutmix alpha value")
    ds_group.add_argument("--random_crop_min", type=float, default=0.08, help="Random crop min value for RRC. Set 1 to disable RRC")
    ds_group.add_argument("--random_erasing_prob", type=float, default=0, help="Random erasing prob, can be used to replace cutout. Set 0 to disable")
    ds_group.add_argument("--rescale_mode", type=str, default="torch", help="Rescale mode, one of [tf, torch]")
    ds_group.add_argument("--eval_central_crop", type=float, default=0.95, help="Evaluation central crop fraction. Set 1 to disable")
    ds_group.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")

    args = parser.parse_known_args(argv)[0]

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
        basic_save_name = "{}_{}_{}_batchsize_{}".format(args.model, args.optimizer, args.data_name, args.batch_size)
        basic_save_name += "_randaug_{}_mixup_{}_cutmix_{}_RRC_{}".format(args.magnitude, args.mixup_alpha, args.cutmix_alpha, args.random_crop_min)
        basic_save_name += "_lr512_{}_wd_{}".format(args.lr_base_512, args.weight_decay)
    args.basic_save_name = basic_save_name

    return args


if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.experimental.get_visible_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.OneDeviceStrategy(device="/gpu:0")

    import sys

    args = parse_arguments(sys.argv[1:])
    print(">>>> ALl args:", args)
    if args.seed is not None:
        print(">>>> Set random seed:", args.seed)
        tf.random.set_seed(args.seed)

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (args.input_shape, args.input_shape, 3)
    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = init_dataset(
        data_name=args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        rescale_mode=args.rescale_mode,
        eval_central_crop=args.eval_central_crop,
        random_crop_min=args.random_crop_min,
        resize_method=args.resize_method,
        random_erasing_prob=args.random_erasing_prob,
        magnitude=args.magnitude,
        num_layers=args.num_layers,
    )

    lr_base = args.lr_base_512 * batch_size / 512
    lr_scheduler, lr_total_epochs = init_lr_scheduler(lr_base, args.lr_decay_steps, args.lr_warmup, args.lr_min, args.lr_decay_on_batch)
    epochs = args.epochs if args.epochs != 0 else lr_total_epochs

    with strategy.scope():
        model = init_model(args.model, input_shape, num_classes, args.pretrained, args.restore_path)
        if model.optimizer is None:
            model = compile_model(model, args.optimizer, lr_base, args.weight_decay, args.bce_threshold, args.label_smoothing)
        print(">>>> basic_save_name =", args.basic_save_name)
        # sys.exit()
        train(model, epochs, train_dataset, test_dataset, args.initial_epoch, lr_scheduler, args.basic_save_name)
