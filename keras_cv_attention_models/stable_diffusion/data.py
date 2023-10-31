import os
import numpy as np


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


def init_diffusion_alpha(num_training_steps=1000, beta_max=0.02):
    beta = np.linspace(0.0001, beta_max, num_training_steps).astype("float32")[:, None, None, None]  # expand to calculation on batch dimension
    alpha_bar = np.cumprod(1.0 - beta, axis=0)
    sqrt_alpha_bar = alpha_bar**0.5
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    return sqrt_alpha_bar, sqrt_one_minus_alpha_bar


def build_torch_dataset(images, labels=None, image_size=512, batch_size=32, num_training_steps=1000, use_horizontal_flip=True):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Normalize, Compose, RandomHorizontalFlip, Resize, InterpolationMode, ToTensor

    use_labels = False if labels is None or len(labels) == 0 else True
    image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = init_diffusion_alpha(num_training_steps)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = torch.from_numpy(sqrt_alpha_bar), torch.from_numpy(sqrt_one_minus_alpha_bar)

    if use_labels:
        labels = [ii + 1 for ii in labels]  # add 1 to labels for skipping 0 as non-conditional
        # to_zero_labels_rate = 1 / max(labels)
        # print(">>>> dataset to_zero_labels_rate:", to_zero_labels_rate)

    class _Dataset_(Dataset):
        def __init__(self, images, labels=None, image_size=512):
            self.images, self.labels = images, labels
            self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            self.transforms = Compose(
                [
                    Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    RandomHorizontalFlip() if use_horizontal_flip else lambda image: image,
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
            images = torch.stack(images)
            labels = torch.tensor(labels)  # if torch.rand(()) > to_zero_labels_rate else torch.zeros([batch_size], dtype=torch.int64)  # [perform worse ???]
        else:
            images = torch.stack(batch)
        timestep = torch.randint(num_training_steps, size=(batch_size,))
        noise = torch.randn_like(images)

        xt = sqrt_alpha_bar[timestep] * images + sqrt_one_minus_alpha_bar[timestep] * noise
        return ((xt, labels, timestep), noise) if use_labels else ((xt, timestep), noise)

    dd = _Dataset_(images, labels, image_size)
    return DataLoader(dd, batch_size=batch_size, collate_fn=diffusion_process, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)


def build_tf_dataset(images, labels=None, image_size=512, batch_size=32, num_training_steps=1000, use_horizontal_flip=True):
    import tensorflow as tf
    from keras_cv_attention_models.imagenet.data import tf_imread

    use_labels = False if labels is None or len(labels) == 0 else True
    image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = init_diffusion_alpha(num_training_steps)
    sqrt_alpha_bar, sqrt_one_minus_alpha_bar = tf.convert_to_tensor(sqrt_alpha_bar), tf.convert_to_tensor(sqrt_one_minus_alpha_bar)
    AUTOTUNE, buffer_size, seed = tf.data.AUTOTUNE, batch_size * 100, None

    if use_labels:
        labels = [ii + 1 for ii in labels]  # add 1 to labels for skipping 0 as non-conditional
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(buffer_size=buffer_size, seed=seed)
        # to_zero_labels_rate = 1 / max(labels)
        # print(">>>> dataset to_zero_labels_rate:", to_zero_labels_rate)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(buffer_size=buffer_size, seed=seed)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(len(images))).with_options(options)

    def image_process(image, label=None):
        image = tf_imread(image)
        image = tf.image.resize(image, image_size, method="bicubic", antialias=True)
        image = tf.image.random_flip_left_right(image) if use_horizontal_flip else image
        image = tf.cast(image, tf.float32)
        image.set_shape([*image_size, 3])
        return (image, label) if use_labels else image

    def diffusion_process(image, label=None):
        timestep = tf.random.uniform([batch_size], 0, num_training_steps, dtype="int64")
        noise = tf.random.normal([batch_size, *image_size, 3])
        image = image / 127.5 - 1
        # labels = tf.cond(tf.random.uniform(()) > to_zero_labels_rate, lambda: label, lambda: tf.zeros([batch_size], dtype=label.dtype))

        xt = tf.gather(sqrt_alpha_bar, timestep) * image + tf.gather(sqrt_one_minus_alpha_bar, timestep) * noise
        return ((xt, label, timestep), noise) if use_labels else ((xt, timestep), noise)

    train_dataset = train_dataset.map(image_process, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).map(diffusion_process, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset
