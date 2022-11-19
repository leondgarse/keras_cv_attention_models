#!/usr/bin/env python3
import os
import json
from keras_cv_attention_models.imagenet import data, train_func, losses


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_name", type=str, default="imagenet2012", help="Dataset name from tensorflow_datasets like imagenet2012 cifar10")
    parser.add_argument("-i", "--input_shape", type=int, default=160, help="Model input shape")
    parser.add_argument(
        "-m", "--model", type=str, default="aotnet.AotNet50", help="Model name in format [sub_dir].[model_name]. Or keras.applications name like MobileNet"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
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
        help="""If build model with pretrained weights. Mostly used is one of [imagenet, imagenet21k]. Or specified h5 file for build model -> restore weights.
                This will drop model optimizer, used for `progressive_train_script.py`. Relatively, `restore_path` is used for restore from break point""",
    )
    parser.add_argument(
        "--additional_model_kwargs", type=str, default=None, help="Json format model kwargs like '{\"drop_connect_rate\": 0.05}'. Note all quote marks"
    )
    parser.add_argument("--seed", type=int, default=None, help="Set random seed if not None")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone, set layer.trainable=False till model GlobalAveragePooling2D layer")
    parser.add_argument("--freeze_norm_layers", action="store_true", help="Set layer.trainable=False for BatchNormalization and LayerNormalization")
    parser.add_argument("--disable_float16", action="store_true", help="Disable mixed_float16 training")
    parser.add_argument("--summary", action="store_true", help="show model summary")
    parser.add_argument(
        "--tensorboard_logs",
        type=str,
        default=None,
        help="TensorBoard logs saving path, default None for disable. Set auto for `logs/{basic_save_name} + _ + timestamp`.",
    )
    parser.add_argument("--TPU", action="store_true", help="Run training on TPU. Will set True for dataset `try_gcs` and `drop_remainder`")

    """ Loss arguments """
    loss_group = parser.add_argument_group("Loss arguments")
    loss_group.add_argument("--label_smoothing", type=float, default=0, help="Loss label smoothing value")
    loss_group.add_argument(
        "--bce_threshold", type=float, default=0.2, help="Value [0, 1) for BCE loss target_threshold, set 1 for using CategoricalCrossentropy"
    )

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
    ds_group.add_argument("--magnitude", type=int, default=6, help="Randaug magnitude value")
    ds_group.add_argument("--num_layers", type=int, default=2, help="Number of randaug applied sequentially to an image. Usually best in [1, 3]")
    ds_group.add_argument("--random_crop_min", type=float, default=0.08, help="Random crop min value for RRC. Set 1 to disable RRC")
    ds_group.add_argument("--mixup_alpha", type=float, default=0.1, help="Mixup alpha value")
    ds_group.add_argument("--cutmix_alpha", type=float, default=1.0, help="Cutmix alpha value")
    ds_group.add_argument("--random_erasing_prob", type=float, default=0, help="Random erasing prob, can be used to replace cutout. Set 0 to disable")
    ds_group.add_argument("--eval_central_crop", type=float, default=0.95, help="Evaluation central crop fraction. Set 1 to disable")
    ds_group.add_argument("--rescale_mode", type=str, default="torch", help="Rescale mode, one of [tf, torch]")
    ds_group.add_argument("--resize_method", type=str, default="bicubic", help="Resize method from tf.image.resize, like [bilinear, bicubic]")
    ds_group.add_argument("--disable_antialias", action="store_true", help="Set use antialias=False for tf.image.resize")
    ds_group.add_argument("--disable_positional_related_ops", action="store_true", help="Set use use_positional_related_ops=False for RandAugment")

    """ Token labeling and distillation parameters """
    dt_group = parser.add_argument_group("Token labeling and distillation arguments")
    dt_group.add_argument("--token_label_file", type=str, default=None, help="Specific token label file path")
    dt_group.add_argument("--token_label_loss_weight", type=float, default=0.5, help="Token label loss weight if `token_label_file` is not None")
    dt_group.add_argument(
        "--teacher_model",
        type=str,
        default=None,
        help="Could be: 1. Saved h5 model path. 2. Model name defined in this repo, format [sub_dir].[model_name] like regnet.RegNetZD8. 3. timm model like timm.models.resmlp_12_224",
    )
    dt_group.add_argument("--teacher_model_pretrained", type=str, default="imagenet", help="Teacher model pretrained weight, if not built from h5")
    dt_group.add_argument("--teacher_model_input_shape", type=int, default=-1, help="Teacher model input_shape, -1 for same with `input_shape`")
    dt_group.add_argument("--distill_temperature", type=float, default=10, help="Temperature for DistillKLDivergenceLoss")
    dt_group.add_argument("--distill_loss_weight", type=float, default=1, help="Distill loss weight if `teacher_model` is not None")

    args = parser.parse_known_args(argv)[0]

    # args.additional_model_kwargs = {"drop_connect_rate": 0.05}
    args.additional_model_kwargs = json.loads(args.additional_model_kwargs) if args.additional_model_kwargs else {}

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
        basic_save_name = "{}_{}_{}_{}_batchsize_{}".format(args.model, args.input_shape, args.optimizer, data_name, args.batch_size)
        basic_save_name += "_randaug_{}_mixup_{}_cutmix_{}_RRC_{}".format(args.magnitude, args.mixup_alpha, args.cutmix_alpha, args.random_crop_min)
        basic_save_name += "_lr512_{}_wd_{}".format(args.lr_base_512, args.weight_decay)
        args.basic_save_name = basic_save_name if args.basic_save_name is None else (basic_save_name + args.basic_save_name)
    args.enable_float16 = not args.disable_float16
    args.tensorboard_logs = None if args.tensorboard_logs is None or args.tensorboard_logs.lower() == "none" else args.tensorboard_logs

    return args


# Wrapper this for reuse in progressive_train_script.py
def run_training_by_args(args):
    print(">>>> ALl args:", args)
    # return None, None, None

    strategy = train_func.init_global_strategy(args.enable_float16, args.seed, args.TPU)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (args.input_shape, args.input_shape)
    use_token_label = False if args.token_label_file is None else True
    use_teacher_model = False if args.teacher_model is None else True
    teacher_model_input_shape = input_shape if args.teacher_model_input_shape == -1 else (args.teacher_model_input_shape, args.teacher_model_input_shape)

    # Init model first, for in case of use_token_label, getting token_label_target_patches
    total_images, num_classes, steps_per_epoch, num_channels = data.init_dataset(args.data_name, batch_size=batch_size, info_only=True)
    input_shape = (*input_shape, num_channels)  # Just in case channel is not 3, like mnist being 1...
    teacher_model_input_shape = (*teacher_model_input_shape, num_channels)  # Just in case channel is not 3, like mnist being 1...
    assert not (num_channels != 3 and args.rescale_mode == "torch")  # "torch" mode mean and std are 3 channels
    with strategy.scope():
        model = args.model if args.restore_path is None else args.restore_path
        model = train_func.init_model(model, input_shape, num_classes, args.pretrained, **args.additional_model_kwargs)
        model = train_func.model_post_process(model, args.freeze_backbone, args.freeze_norm_layers, use_token_label)
        if args.summary:
            model.summary()

        if use_teacher_model:
            print(">>>> [Build teacher model]")
            teacher_model = train_func.init_model(
                args.teacher_model, teacher_model_input_shape, num_classes, args.teacher_model_pretrained, reload_compile=False
            )
            model, teacher_model = train_func.init_distill_model(model, teacher_model)
        else:
            teacher_model = None
    token_label_target_patches = model.output_shape[-1][1:-1] if use_token_label else -1

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        rescale_mode=args.rescale_mode,
        eval_central_crop=args.eval_central_crop,
        random_crop_min=args.random_crop_min,
        resize_method=args.resize_method,
        resize_antialias=not args.disable_antialias,
        random_erasing_prob=args.random_erasing_prob,
        magnitude=args.magnitude,
        num_layers=args.num_layers,
        use_positional_related_ops=not args.disable_positional_related_ops,
        token_label_file=args.token_label_file,
        token_label_target_patches=token_label_target_patches,
        teacher_model=teacher_model,
        teacher_model_input_shape=teacher_model_input_shape,
    )

    lr_base = args.lr_base_512 * batch_size / 512
    warmup_steps, cooldown_steps, t_mul, m_mul = args.lr_warmup_steps, args.lr_cooldown_steps, args.lr_t_mul, args.lr_m_mul  # Save line-width
    lr_scheduler, lr_total_epochs = train_func.init_lr_scheduler(
        lr_base, args.lr_decay_steps, args.lr_min, args.lr_decay_on_batch, args.lr_warmup, warmup_steps, cooldown_steps, t_mul, m_mul
    )
    epochs = args.epochs if args.epochs != -1 else lr_total_epochs

    with strategy.scope():
        token_label_loss_weight = args.token_label_loss_weight if use_token_label else 0
        distill_loss_weight = args.distill_loss_weight if use_teacher_model else 0
        loss, loss_weights, metrics = train_func.init_loss(
            args.bce_threshold, args.label_smoothing, token_label_loss_weight, distill_loss_weight, args.distill_temperature, model.output_names
        )

        if model.optimizer is None:
            # optimizer can be a str like "sgd" / "adamw" / "lamb", or specific initialized `keras.optimizers.xxx` instance.
            # Or just call `model.compile(...)` by self.
            model = train_func.compile_model(model, args.optimizer, lr_base, args.weight_decay, loss, loss_weights, metrics, args.momentum)
        print(">>>> basic_save_name =", args.basic_save_name)
        # return None, None, None
        latest_save, hist = train_func.train(
            model, epochs, train_dataset, test_dataset, args.initial_epoch, lr_scheduler, args.basic_save_name, logs=args.tensorboard_logs
        )
    return model, latest_save, hist


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    run_training_by_args(args)
