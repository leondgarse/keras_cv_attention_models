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
    def __init__(self, save_path, image_size=512, num_classes=0, num_training_steps=1000, num_steps=50, rows=4, cols=4):
        super().__init__()
        self.save_path, self.image_size, self.num_training_steps, self.num_steps = save_path, image_size, num_training_steps, num_steps
        self.num_classes, self.rows, self.cols, self.batch_size = num_classes, rows, cols, rows * cols
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if kecam.backend.image_data_format() == "channels_last":
            self.noise_shape = (self.batch_size, image_size, image_size, 3)
            self.is_channels_last = True
        else:
            self.noise_shape = (self.batch_size, 3, image_size, image_size)
            self.is_channels_last = False

        self.eval_x0 = np.random.normal(size=self.noise_shape).astype("float32")
        if self.num_classes > 0:
            self.labels_inputs = np.random.uniform(0, self.num_classes, size=[self.batch_size]).astype("int64")

        self.timesteps = np.arange(0, num_training_steps, num_training_steps // num_steps)[::-1] + 1
        # self.timesteps = np.stack([timesteps] * self.batch_size, axis=-1)[:, :, None, None, None]  # -> [num_steps, batch_size, None, None, None]
        self.beta = np.linspace(0.0001, 0.02, num_training_steps).astype("float32")
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha, axis=0)
        self.eps_coef = (1 - self.alpha) / (1 - self.alpha_bar) ** 0.5

        self.device = None  # Set to actual torch model using device later
        self.to_device = (lambda xx: xx.to(self.device)) if kecam.backend.is_torch_backend else (lambda xx: xx)
        self.to_host = (lambda xx: xx.cpu()) if kecam.backend.is_torch_backend else (lambda xx: xx)

    def on_epoch_end(self, cur_epoch=0, logs=None):
        if kecam.backend.is_torch_backend and self.device is None:
            self.device = next(self.model.parameters()).device
        compute_dtype = self.model.compute_dtype
        xt = functional.convert_to_tensor(self.eval_x0, dtype=compute_dtype)
        xt = self.to_device(xt)
        if self.num_classes > 0:
            labels_inputs = self.to_device(functional.convert_to_tensor(self.labels_inputs, dtype="int64"))

        for timestep in self.timesteps:
            timestep_inputs = functional.convert_to_tensor(np.stack([timestep] * self.batch_size), dtype=compute_dtype)
            timestep_inputs = self.to_device(timestep_inputs)
            xt_noise = self.model([xt, labels_inputs, timestep_inputs]) if self.num_classes > 0 else self.model([xt, timestep_inputs])

            eps = functional.convert_to_tensor(np.random.normal(size=self.noise_shape).astype("float32"), dtype=compute_dtype)
            cur_alpha, cur_eps_coef = self.alpha[timestep], self.eps_coef[timestep]
            xt = 1 / (cur_alpha**0.5) * (xt - cur_eps_coef * xt_noise) + ((1 - cur_alpha) ** 0.5) * self.to_device(eps)

        xt = self.to_host(xt).numpy()
        eval_xt = xt if self.is_channels_last else xt.transpose([0, 2, 3, 1])
        eval_xt = np.vstack([np.hstack(eval_xt[row * self.cols : (row + 1) * self.cols]) for row in range(self.rows)])
        eval_xt = (np.clip(eval_xt / 2 + 0.5, 0, 1) * 255).astype("uint8")

        save_path = os.path.join(self.save_path, "epoch_{}.jpg".format(cur_epoch + 1))
        Image.fromarray(eval_xt).save(save_path)
        print(">>>> Epoch {} image saved to {}".format(cur_epoch + 1, save_path))


class DefussionDatasetGen:
    def __init__(self, data_path, image_size=512, batch_size=32, num_training_steps=1000):
        self.data_path, self.image_size, self.batch_size, self.num_training_steps = data_path, image_size, batch_size, num_training_steps
        if data_path.endswith(".json"):
            self.all_images, self.all_labels, self.num_classes = self.init_from_json()
            print(">>>> total images found: {}, num_classes: {}".format(len(self.all_images), self.num_classes))
        else:
            self.all_images, self.all_labels, self.num_classes = self.walk_data_path_gather_images(), [], 0
            print(">>>> total images found:", len(self.all_images))

        self.total = len(self.all_images)
        self.steps_per_epoch = self.total // batch_size  # Drop remaining
        if kecam.backend.image_data_format() == "channels_last":
            self.noise_shape = (batch_size, image_size, image_size, 3)
            self.is_channels_last = True
        else:
            self.noise_shape = (batch_size, 3, image_size, image_size)
            self.is_channels_last = False

        self.beta = np.linspace(0.0001, 0.02, num_training_steps).astype("float32")[:, None, None, None]  # expand to calculation on batch dimension
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha, axis=0)

    def walk_data_path_gather_images(self):
        all_images = []
        for cur, dirs, files in os.walk(self.data_path):
            all_images.extend([os.path.join(cur, ii) for ii in files if os.path.splitext(ii)[-1].lower() in [".jpg", ".png"]])
        return np.array(all_images)

    def init_from_json(self):
        import json

        with open(self.data_path, "r") as ff:
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
        return np.array(all_images), np.array(all_labels), num_classes

    def imread(self, image_path):
        return np.array(Image.open(image_path).convert("RGB").resize([self.image_size, self.image_size], resample=Image.Resampling.BICUBIC)).astype("float32")

    def __iter__(self):
        self.generated = 0
        # print(">>>> Data shuffle")
        if self.num_classes > 0:
            shuffle_indexes = np.random.permutation(self.all_images.shape[0])
            self.all_images = self.all_images[shuffle_indexes]
            self.all_labels = self.all_labels[shuffle_indexes]
        else:
            self.all_images = np.random.permutation(self.all_images)
        return self

    def __len__(self):
        return self.steps_per_epoch

    def __next__(self):
        if self.generated < self.steps_per_epoch:
            cur_batch = self.generated * self.batch_size
            self.generated += 1

            images = self.all_images[cur_batch : cur_batch + self.batch_size]
            images = np.stack([self.imread(ii) for ii in images])
            images = images / 127.5 - 1  # [0, 255] -> [-1, 1]
            images = images if np.random.uniform() > 0.5 else np.flip(images, axis=2)  # Random flip left right
            images = images if self.is_channels_last else images.transpose([0, 3, 1, 2])

            # diffusion process
            timestep = np.random.uniform(0, self.num_training_steps, [self.batch_size]).astype("int64")
            noise = np.random.normal(size=self.noise_shape).astype("float32")

            cur_alpha = self.alpha_bar[timestep]
            xt = cur_alpha**0.5 * images + (1 - cur_alpha) ** 0.5 * noise  # Sample from $q(x_t|x_0)$

            if self.num_classes > 0:
                return ((xt, self.all_labels[cur_batch : cur_batch + self.batch_size], timestep.astype("float32")), noise)
            else:
                return (xt, timestep.astype("float32")), noise
        else:
            raise StopIteration

    def __getitem__(self, idx):
        return self.__next__()


def build_torch_dataset(dataset_gen):
    from torch.utils.data import DataLoader, IterableDataset

    class DD(IterableDataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            return iter(self.dataset)

    train_dataset = DataLoader(DD(dataset_gen), batch_size=None)
    return train_dataset


def build_tf_dataset(dataset_gen):
    image_signature = tf.TensorSpec(shape=(None, dataset_gen.image_size, dataset_gen.image_size, 3), dtype=tf.float32)
    if dataset_gen.num_classes > 0:  # With labels as inputs, [(images, labels, timesteps), noise]
        output_signature = ((image_signature, tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None,), dtype=tf.float32)), image_signature)
    else:  # [(images, timesteps), noise]
        output_signature = ((image_signature, tf.TensorSpec(shape=(None,), dtype=tf.int64)), image_signature)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = tf.data.Dataset.from_generator(lambda: iter(dataset_gen), output_signature=output_signature)
    train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(dataset_gen.steps_per_epoch)).with_options(options)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
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
    parser.add_argument("--num_eval_plot", type=int, default=16, help="number of eval plot images, will take less than `batch_size`")
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")

    parser.add_argument("--lr_base_512", type=float, default=1e-3, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    parser.add_argument("--lr_warmup_steps", type=int, default=3, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
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

    train_dataset_gen = DefussionDatasetGen(args.data_path, args.input_shape, batch_size=args.batch_size, num_training_steps=args.num_training_steps)
    inputs, noise = next(iter(train_dataset_gen))
    print(">>>> Total train batches: {}".format(len(train_dataset_gen)))
    print(">>>> Data: inputs: {}, noise.shape: {}".format([ii.shape for ii in inputs], noise.shape))
    num_classes = train_dataset_gen.num_classes
    train_dataset = build_torch_dataset(train_dataset_gen) if kecam.backend.is_torch_backend else build_tf_dataset(train_dataset_gen)

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

        lr_scheduler = kecam.imagenet.callbacks.CosineLrScheduler(lr, args.epochs, steps_per_epoch=len(train_dataset), warmup_steps=args.lr_warmup_steps)
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
