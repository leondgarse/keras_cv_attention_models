import os
import numpy as np
import kecam
from PIL import Image
from kecam.backend import functional

BUILDIN_DATASETS = {
    "coco_dog_cat": {
        "url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/coco_dog_cat.tar.gz",
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


class DenoisingEval(kecam.backend.callbacks.Callback):
    def __init__(self, save_path, image_size=512, num_classes=0, num_training_steps=1000, num_steps=-1, rows=5, cols=4):
        super().__init__()
        self.save_path, self.image_size, self.num_training_steps = save_path, image_size, num_training_steps
        self.num_classes, self.rows, self.cols, self.batch_size = num_classes, rows, cols, rows * cols
        self.num_steps = num_steps if num_steps > 0 else num_training_steps
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if kecam.backend.image_data_format() == "channels_last":
            self.noise_shape = (self.batch_size, image_size, image_size, 3)
            self.is_channels_last = True
        else:
            self.noise_shape = (self.batch_size, 3, image_size, image_size)
            self.is_channels_last = False

        self.device = None  # Set to actual torch model using device later
        self.to_device = (lambda xx: xx.to(self.device)) if kecam.backend.is_torch_backend else (lambda xx: xx)
        self.to_host = (lambda xx: xx.cpu()) if kecam.backend.is_torch_backend else (lambda xx: xx)

        self.eval_x0 = np.random.normal(size=self.noise_shape).astype("float32")
        if self.num_classes > 0:
            self.labels_inputs = np.random.uniform(0, self.num_classes, size=[self.batch_size]).astype("int64")
            labels_inputs = np.arange(0, self.num_classes, self.num_classes / self.batch_size)[: self.batch_size].astype("int64")
            self.labels_inputs = np.pad(labels_inputs, [0, self.batch_size - labels_inputs.shape[0]], constant_values=self.num_classes - 1)  # Just in case
            print(">>>> Eval labels:", self.labels_inputs)

        self.timesteps = np.arange(0, num_training_steps, num_training_steps // self.num_steps).astype("int64")
        beta = np.linspace(0.0001, 0.02, num_training_steps).astype("float32")[self.timesteps]
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha, axis=0)
        self.xt_prev_alpha = 1 / (alpha**0.5)
        self.xt_noise_alpha = self.xt_prev_alpha * (1 - alpha) / (1 - alpha_bar) ** 0.5
        self.eps_alpha = beta**0.5

    def on_epoch_end(self, cur_epoch=0, logs=None):
        if kecam.backend.is_torch_backend and self.device is None:
            self.device = next(self.model.parameters()).device
        compute_dtype = self.model.compute_dtype
        xt = functional.convert_to_tensor(self.eval_x0, dtype=compute_dtype)
        xt = self.to_device(xt)
        if self.num_classes > 0:
            labels_inputs = self.to_device(functional.convert_to_tensor(self.labels_inputs, dtype="int64"))

        for cur_step in range(self.num_steps)[::-1]:
            timestep_inputs = functional.convert_to_tensor(np.stack([self.timesteps[cur_step]] * self.batch_size), dtype="int64")
            timestep_inputs = self.to_device(timestep_inputs)
            xt_noise = self.model([xt, labels_inputs, timestep_inputs]) if self.num_classes > 0 else self.model([xt, timestep_inputs])

            eps = functional.convert_to_tensor(np.random.normal(size=self.noise_shape).astype("float32"), dtype=compute_dtype)
            xt = self.xt_prev_alpha[cur_step] * xt - self.xt_noise_alpha[cur_step] * xt_noise + self.eps_alpha[cur_step] * self.to_device(eps)

        xt = self.to_host(xt).numpy()
        eval_xt = xt if self.is_channels_last else xt.transpose([0, 2, 3, 1])
        eval_xt = np.vstack([np.hstack(eval_xt[row * self.cols : (row + 1) * self.cols]) for row in range(self.rows)])
        eval_xt = (np.clip(eval_xt / 2 + 0.5, 0, 1) * 255).astype("uint8")

        save_path = os.path.join(self.save_path, "epoch_{}.jpg".format(cur_epoch + 1))
        Image.fromarray(eval_xt).save(save_path)
        print(">>>> Epoch {} image saved to {}".format(cur_epoch + 1, save_path))


def walk_data_path_gather_images(data_path):
    all_images = []
    for cur, dirs, files in os.walk(data_path):
        all_images.extend([os.path.join(cur, ii) for ii in files if os.path.splitext(ii)[-1].lower() in [".jpg", ".png"]])
    return all_images


def init_from_json(data_path):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)
    train, info = aa["train"], aa.get("info", {})

    all_images = [ii["image"] for ii in train]
    all_labels = [ii["label"] for ii in train] if "label" in train[0] else []
    total_images, num_classes = len(train), info.get("num_classes", 0)

    if num_classes <= 0 and "label" in train[0] and isinstance(train[0]["label"], int):
        num_classes = max(all_labels) + 1
        print(">>>> Using max value from train as num_classes:", num_classes)
    if "base_path" in info and len(info["base_path"]) > 0:
        base_path = os.path.expanduser(info["base_path"])
        all_images = [os.path.join(base_path, ii) for ii in all_images]
    return all_images, all_labels, num_classes


def init_diffusion_alpha(num_training_steps=1000):
    beta = np.linspace(0.0001, 0.02, num_training_steps).astype("float32")[:, None, None, None]  # expand to calculation on batch dimension
    alpha_bar = np.cumprod(1.0 - beta, axis=0)
    sqrt_alpha_bar = alpha_bar**0.5
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    return sqrt_alpha_bar, sqrt_one_minus_alpha_bar


def build_torch_dataset(images, labels=None, image_size=512, batch_size=32, num_training_steps=1000):
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Normalize, Compose, RandomHorizontalFlip, Resize, InterpolationMode, ToTensor

    use_labels = False if labels is None or len(labels) == 0 else True
    image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = init_diffusion_alpha(num_training_steps)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = torch.from_numpy(sqrt_alpha_bar), torch.from_numpy(sqrt_one_minus_alpha_bar)

    class _Dataset_(Dataset):
        def __init__(self, images, labels=None, image_size=512):
            self.images, self.labels = images, labels
            self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            self.transforms = Compose(
                [
                    Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    RandomHorizontalFlip(),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std),
                ]
            )

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.transforms(Image.open(str(self.images[idx])))
            return (image, self.labels[idx]) if use_labels else image

    def diffusion_process(batch):
        if use_labels:
            images, labels = list(zip(*batch))
            images, labels = torch.stack(images), torch.tensor(labels)
        else:
            images = torch.stack(batch)
        timestep = torch.randint(num_training_steps, size=(batch_size,))
        noise = torch.randn_like(images)

        xt = sqrt_alpha_bar[timestep] * images + sqrt_one_minus_alpha_bar[timestep] * noise
        return ((xt, labels, timestep), noise) if use_labels else ((xt, timestep), noise)

    dd = _Dataset_(images, labels, image_size)
    return DataLoader(dd, batch_size=batch_size, collate_fn=diffusion_process, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)


def build_tf_dataset(images, labels=None, image_size=512, batch_size=32, num_training_steps=1000):
    from keras_cv_attention_models.imagenet.data import tf_imread

    use_labels = False if labels is None or len(labels) == 0 else True
    image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = init_diffusion_alpha(num_training_steps)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = tf.convert_to_tensor(sqrt_alpha_bar), tf.convert_to_tensor(sqrt_one_minus_alpha_bar)
    AUTOTUNE, buffer_size, seed = tf.data.AUTOTUNE, batch_size * 100, None

    if use_labels:
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(buffer_size=len(images), seed=seed)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(buffer_size=len(images), seed=seed)

    def image_process(image, label=None):
        image = tf_imread(image)
        image = tf.image.resize(image, image_size, method="bicubic", antialias=True)
        image = tf.cast(image, tf.float32)
        image.set_shape([*image_size, 3])
        return (image, label) if use_labels else image

    def diffusion_process(image, label=None):
        timestep = tf.random.uniform([batch_size], 0, num_training_steps, dtype="int64")
        noise = tf.random.normal([batch_size, *image_size, 3])

        xt = tf.gather(sqrt_alpha_bar, timestep) * image + tf.gather(sqrt_one_minus_alpha_bar, timestep) * noise
        return ((xt, label, timestep), noise) if use_labels else ((xt, timestep), noise)

    train_dataset = train_dataset.map(image_process, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).map(diffusion_process, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset


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

    optimizer = tf.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, beta_1=beta1, beta_2=beta2, epsilon=eps)
    optimizer.exclude_from_weight_decay(var_names=no_weight_decay)
    return optimizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="coco_dog_cat",
        help="dataset directory path containing images, or a recognition json dataset path, which will train using labels as instruction",
    )
    parser.add_argument("-i", "--input_shape", type=int, default=512, help="Model input shape")
    parser.add_argument("-m", "--model", type=str, default="UNet", help="model from this repo `[model_classs.model_name]` like stable_diffusion.UNet")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="training epochs, total max iterations=epochs * steps_per_epoch")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default=None, help="Basic save name for model and history")
    parser.add_argument("-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 or pt file. Higher priority than model")
    parser.add_argument("--num_training_steps", type=int, default=1000, help="train sampling steps")
    parser.add_argument("--num_eval_plot", type=int, default=20, help="number of eval plot images, will take less than `batch_size`")
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")

    parser.add_argument("--lr_base_512", type=float, default=1e-3, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    parser.add_argument("--lr_warmup_steps", type=int, default=3, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_known_args()[0]
    if args.basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
        args.basic_save_name = basic_save_name
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.data_path in BUILDIN_DATASETS and not os.path.exists(args.data_path):
        url, dataset_file = BUILDIN_DATASETS[args.data_path]["url"], BUILDIN_DATASETS[args.data_path]["dataset_file"]
        file_path = kecam.backend.get_file(origin=url, cache_subdir="datasets", extract=True)  # returned tar file path
        args.data_path = os.path.join(os.path.dirname(file_path), args.data_path, dataset_file)
        print(">>>> Buildin dataset, path:", args.data_path)

    if args.data_path.endswith(".json"):
        all_images, all_labels, num_classes = init_from_json(args.data_path)
        print(">>>> total images found: {}, num_classes: {}".format(len(all_images), num_classes))
    else:
        all_images, all_labels, num_classes = walk_data_path_gather_images(args.data_path), None, 0
        print(">>>> total images found:", len(all_images))

    if kecam.backend.is_torch_backend:
        train_dataset = build_torch_dataset(all_images, all_labels, args.input_shape, batch_size=args.batch_size, num_training_steps=args.num_training_steps)
    else:
        train_dataset = build_tf_dataset(all_images, all_labels, args.input_shape, batch_size=args.batch_size, num_training_steps=args.num_training_steps)

    inputs, noise = next(iter(train_dataset))
    print(">>>> Total train batches: {}".format(len(train_dataset)))
    print(">>>> Data: inputs: {}, noise.shape: {}".format([ii.shape for ii in inputs], noise.shape))

    lr = args.lr_base_512 * args.batch_size / 512
    print(">>>> lr:", lr)

    with global_strategy.scope():
        if args.restore_path is None or kecam.backend.is_torch_backend:
            model_split = args.model.split(".")
            model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
            kwargs = {} if args.pretrained == "default" else {"pretrained": args.pretrained}
            # Not using prompt conditional inputs, but may use labels as inputs if num_classes > 0
            kwargs.update({"conditional_embedding": 0, "input_shape": inputs[0].shape[1:], "num_classes": num_classes})
            print(">>>> model_kwargs:", kwargs)
            model = model_class(**kwargs)
            print(">>>> model name: {}, input_shape: {}, output_shape: {}".format(model.name, model.input_shape, model.output_shape))
            basic_save_name = args.basic_save_name or "ddpm_{}_{}".format(model.name, kecam.backend.backend())
        else:
            print(">>>> Reload from:", args.restore_path)
            model = kecam.backend.models.load_model(args.restore_path)
        print(">>>> basic_save_name:", basic_save_name)

        if kecam.backend.is_torch_backend:
            model.to(device=global_device)
            optimizer = build_torch_optimizer(model, lr=lr, weight_decay=args.weight_decay)
            if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
                print(">>>> Calling torch.compile")
                model = torch.compile(model)
            model.train_compile(optimizer=optimizer, loss=kecam.backend.losses.MeanSquaredError())  # `compile` is took by `nn.Module`

            if args.restore_path is not None:
                print(">>>> Reload weights from:", args.restore_path)
                model.load(args.restore_path)  # Reload wights after compile
        elif model.optimizer is None:
            optimizer = build_tf_optimizer(lr=lr, weight_decay=args.weight_decay)
            model.compile(optimizer=optimizer, loss=kecam.backend.losses.MeanSquaredError())

        # Save an image to `checkpoints/{basic_save_name}/epoch_{id}.jpg` on each epoch end
        cols, rows = kecam.plot_func.get_plot_cols_rows(min(args.batch_size, args.num_eval_plot))  # 16 -> [4, 4]; 42 -> [7, 6]
        save_path = os.path.join("checkpoints", basic_save_name)
        eval_callback = DenoisingEval(save_path, args.input_shape, num_classes, args.num_training_steps, cols=cols, rows=rows)

        lr_scheduler = kecam.imagenet.callbacks.CosineLrSchedulerEpoch(lr, args.epochs, lr_warmup=1e-4, warmup_steps=args.lr_warmup_steps)
        other_kwargs = {}
        latest_save, hist = kecam.imagenet.train_func.train(
            compiled_model=model,
            epochs=args.epochs,
            train_dataset=train_dataset,
            test_dataset=None,
            initial_epoch=args.initial_epoch,
            lr_scheduler=lr_scheduler,
            basic_save_name=basic_save_name,
            init_callbacks=[eval_callback],
            logs=None,
            **other_kwargs,
        )
