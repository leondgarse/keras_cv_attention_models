import os
import json
import kecam


BUILDIN_DATASETS = {
    "cifar10": {
        "url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/cifar10.tar.gz",
        "dataset_file": "recognition.json",
    },
    "coco_dog_cat": {
        "url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/coco_dog_cat.tar.gz",
        "dataset_file": "recognition.json",
    },
    "gtsrb": {
        "url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/gtsrb.tar.gz",
        "dataset_file": "recognition.json",
    },
}

if kecam.backend.is_torch_backend:  # os.environ["KECAM_BACKEND"] = "torch"
    import torch
    from collections import namedtuple
    from contextlib import nullcontext

    global_strategy = namedtuple("strategy", ["scope"])(nullcontext)  # Fake
    # Always 0, no matter CUDA_VISIBLE_DEVICES
    global_device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
else:
    import tensorflow as tf
    from keras_cv_attention_models.imagenet.train_func import init_global_strategy

    global_strategy = init_global_strategy(enable_float16=len(tf.config.experimental.get_visible_devices("GPU")) > 0)


def build_torch_optimizer(model, lr=1e-4, weight_decay=0.2, beta1=0.9, beta2=0.95, eps=1.0e-6):
    import inspect

    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["gamma", "beta", "bias", "positional_embedding", "no_weight_decay"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": weight_decay},
    ]

    optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer


def build_tf_optimizer(lr=1e-4, weight_decay=0.2, beta1=0.9, beta2=0.95, eps=1.0e-6):
    no_weight_decay = ["/gamma", "/beta", "/bias", "/positional_embedding", "/no_weight_decay"]

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, beta_1=beta1, beta_2=beta2, epsilon=eps)
    optimizer.exclude_from_weight_decay(var_names=no_weight_decay)
    return optimizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="cifar10",
        help="dataset directory path containing images, or a recognition json dataset path, which will train using labels as instruction",
    )
    parser.add_argument("-i", "--input_shape", type=int, default=32, help="Model input shape")
    parser.add_argument("-m", "--model", type=str, default="UNetTest", help="model from this repo `[model_classs.model_name]` like stable_diffusion.UNet")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="training epochs, total max iterations=epochs * steps_per_epoch")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default=None, help="Basic save name for model and history")
    parser.add_argument("-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 or pt file. Higher priority than model")
    parser.add_argument("--num_training_steps", type=int, default=1000, help="train sampling steps")
    parser.add_argument("--num_eval_plot", type=int, default=20, help="number of eval plot images, will take less than `batch_size`")
    parser.add_argument("--eval_interval", type=int, default=10, help="number of epochs interval running eval process")
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")
    parser.add_argument(
        "--additional_model_kwargs", type=str, default=None, help="Json format model kwargs like '{\"dropout\": 0.15}'. Note all quote marks"
    )

    parser.add_argument("--lr_base_512", type=float, default=1e-3, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    parser.add_argument("--lr_warmup_steps", type=float, default=0.1, help="Learning rate warmup steps, <1 for `lr_warmup_steps * epochs`, >=1 for exact value")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--disable_horizontal_flip", action="store_true", help="Disable random horizontal flip")
    args = parser.parse_known_args()[0]

    args.additional_model_kwargs = json.loads(args.additional_model_kwargs) if args.additional_model_kwargs else {}
    if args.basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
        args.basic_save_name = basic_save_name
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(">>>> args:", args)

    if args.data_path in BUILDIN_DATASETS and not os.path.exists(args.data_path):
        url, dataset_file = BUILDIN_DATASETS[args.data_path]["url"], BUILDIN_DATASETS[args.data_path]["dataset_file"]
        file_path = os.path.join(os.path.expanduser("~"), ".keras", "datasets", args.data_path)
        if not os.path.exists(file_path):
            file_path = kecam.backend.get_file(origin=url, cache_subdir="datasets", extract=True)  # returned tar file path
        args.data_path = os.path.join(os.path.dirname(file_path), args.data_path, dataset_file)
        print(">>>> Buildin dataset, path:", args.data_path)

    if args.data_path.endswith(".json"):
        all_images, all_labels, num_classes = kecam.stable_diffusion.data.init_from_json(args.data_path)
        print(">>>> total images found: {}, num_classes: {}".format(len(all_images), num_classes))
    else:
        all_images, all_labels, num_classes = kecam.stable_diffusion.data.walk_data_path_gather_images(args.data_path), None, 0
        print(">>>> total images found:", len(all_images))

    use_horizontal_flip = not args.disable_horizontal_flip
    build_dataset = kecam.stable_diffusion.data.build_torch_dataset if kecam.backend.is_torch_backend else kecam.stable_diffusion.data.build_tf_dataset
    train_dataset = build_dataset(all_images, all_labels, args.input_shape, args.batch_size, args.num_training_steps, use_horizontal_flip=use_horizontal_flip)

    inputs, noise = next(iter(train_dataset))
    print(">>>> Total train batches: {}".format(len(train_dataset)))
    print(">>>> Data: inputs: {}, noise.shape: {}".format([ii.shape for ii in inputs], noise.shape))

    lr = args.lr_base_512 * args.batch_size / 512
    lr_warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps >= 1 else int(args.lr_warmup_steps * args.epochs)
    lr_cooldown_steps = 5
    print(">>>> lr: {}, lr_warmup_steps: {}, lr_cooldown_steps: {}".format(lr, lr_warmup_steps, lr_cooldown_steps))

    with global_strategy.scope():
        if args.restore_path is None or kecam.backend.is_torch_backend:
            model_split = args.model.split(".")
            model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
            if args.pretrained != "default":
                args.additional_model_kwargs.update({"pretrained": args.pretrained})
            # Not using prompt conditional inputs, but may use labels as inputs if num_classes > 0
            args.additional_model_kwargs.update({"conditional_embedding": 0, "input_shape": inputs[0].shape[1:], "num_classes": num_classes})
            print(">>>> model_kwargs:", args.additional_model_kwargs)
            model = model_class(**args.additional_model_kwargs)
            print(">>>> model name: {}, input_shape: {}, output_shape: {}".format(model.name, model.input_shape, model.output_shape))
            args.basic_save_name = args.basic_save_name or "ddpm_{}_{}".format(model.name, kecam.backend.backend())
        else:
            print(">>>> Reload from:", args.restore_path)
            model = kecam.backend.models.load_model(args.restore_path, custom_objects={"AdamW": tf.keras.optimizers.AdamW})
        print(">>>> basic_save_name:", args.basic_save_name)

        if kecam.backend.is_torch_backend:
            model.to(device=global_device)
            optimizer = build_torch_optimizer(model, lr=lr, weight_decay=args.weight_decay)
            if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
                print(">>>> Calling torch.compile")
                model = torch.compile(model)
            # loss_weights = float(np.prod(inputs[0].shape[1:])) / args.num_training_steps
            model.train_compile(optimizer=optimizer, loss=kecam.backend.losses.MeanSquaredError())  # , loss_weights=loss_weights)

            if args.restore_path is not None:
                print(">>>> Reload weights from:", args.restore_path)
                model.load(args.restore_path)  # Reload wights after compile
        elif model.optimizer is None:
            optimizer = build_tf_optimizer(lr=lr, weight_decay=args.weight_decay)
            model.compile(optimizer=optimizer, loss=kecam.backend.losses.MeanSquaredError())

        # Save an image to `checkpoints/{basic_save_name}/epoch_{id}.jpg` on each epoch end
        cols, rows = kecam.plot_func.get_plot_cols_rows(min(args.batch_size, args.num_eval_plot))  # 16 -> [4, 4]; 42 -> [7, 6]
        save_path = os.path.join("checkpoints", args.basic_save_name)
        eval_callback = kecam.stable_diffusion.eval_func.DenoisingEval(
            save_path, args.input_shape, num_classes, args.num_training_steps, interval=args.eval_interval, cols=cols, rows=rows
        )

        lr_scheduler = kecam.imagenet.callbacks.CosineLrSchedulerEpoch(
            lr, args.epochs, lr_warmup=1e-4, warmup_steps=lr_warmup_steps, cooldown_steps=lr_cooldown_steps
        )
        other_kwargs = {}
        latest_save, hist = kecam.imagenet.train_func.train(
            compiled_model=model,
            epochs=args.epochs + lr_cooldown_steps,
            train_dataset=train_dataset,
            test_dataset=None,
            initial_epoch=args.initial_epoch,
            lr_scheduler=lr_scheduler,
            basic_save_name=args.basic_save_name,
            init_callbacks=[eval_callback],  # if kecam.backend.is_torch_backend else [],  # [???] TF backend prediction runs rather slow
            logs=None,
            **other_kwargs,
        )
