import os
import math
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from keras_cv_attention_models.common_layers import init_mean_std_by_rescale_mode
from keras_cv_attention_models.plot_func import show_batch_sample


INTERPOLATION_MODE_MAP = {"nearest": "NEAREST", "bilinear": "BILINEAR", "bicubic": "BICUBIC", "area": "BOX"}


def load_from_custom_json(data_path):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)
    test_key = "validation" if "validation" in aa else "test"
    train, test, info = aa["train"], aa[test_key], aa.get("info", {})

    total_images, num_classes = len(train), info.get("num_classes", 0)
    if num_classes <= 0:
        num_classes = max([ii["label"] for ii in train]) + 1
        print(">>>> Using max value from train as num_classes:", num_classes)

    if "base_path" in info and len(info["base_path"]) > 0:
        base_path = os.path.expanduser(info["base_path"])
        for ii in train:
            ii["image"] = os.path.join(base_path, ii["image"])
        for ii in test:
            ii["image"] = os.path.join(base_path, ii["image"])

    train_images, train_labels = [ii["image"] for ii in train], [ii["label"] for ii in train]
    test_images, test_labels = [ii["image"] for ii in test], [ii["label"] for ii in test]
    return train_images, train_labels, test_images, test_labels, total_images, num_classes


def image_center_crop(image, central_crop=1.0):
    width, height = image.size
    croped_size = int(central_crop * float(min(height, width)))
    yy, xx = (height - croped_size) // 2, (width - croped_size) // 2
    return image.crop([xx, yy, xx + croped_size, yy + croped_size])  # left, upper, right, lower


class RecognitionDatset(Dataset):
    def __init__(self, images, labels, num_classes, is_train=False, input_shape=(224, 224), central_crop=1.0, rescale_mode="torch", resize_method="bilinear"):
        from torchvision.transforms import Normalize, Compose, RandomResizedCrop, CenterCrop, Resize, InterpolationMode, ToTensor

        self.images, self.labels, self.num_classes = images, labels, num_classes
        mean, std = init_mean_std_by_rescale_mode(rescale_mode)
        self.mean, self.std = mean / 255, std / 255  # ToTensor is already converted / 255
        interpolation = getattr(InterpolationMode, INTERPOLATION_MODE_MAP.get(resize_method, "INTER_LINEAR"))
        input_shape = input_shape if isinstance(input_shape, (list, tuple)) else (input_shape, input_shape)
        self.transforms = Compose(
            [
                lambda image: image if is_train else image_center_crop(image, central_crop=central_crop),
                RandomResizedCrop(input_shape, scale=(0.9, 1.0), interpolation=interpolation) if is_train else Resize(input_shape, interpolation=interpolation),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])))
        label = torch.functional.F.one_hot(torch.tensor(self.labels[idx]), self.num_classes)
        return image, label


def init_dataset(
    data_path,
    input_shape=(224, 224),
    batch_size=64,
    rescale_mode="tf",
    eval_central_crop=1.0,
    resize_method="bilinear",
    resize_antialias=False,
    num_workers=8,
    max_mosaic_cache_len=1024,
    info_only=False,
):
    """
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> from keras_cv_attention_models.imagenet import data
    >>> from keras_cv_attention_models.plot_func import show_recognition_batch_sample
    >>> train, test = data.init_dataset('datasets/coco_dog_cat/recognition.json')[:2]
    >>> ax = data.show_batch_sample(test, indices_2_labels={0: 'cat', 1: 'dog'})
    >>> ax.get_figure().savefig('aa.jpg')
    """
    train_images, train_labels, test_images, test_labels, total_images, num_classes = load_from_custom_json(data_path)
    steps_per_epoch = int(math.ceil(total_images / float(batch_size)))

    if info_only:
        num_channels = 3
        return total_images, num_classes, steps_per_epoch, num_channels  # return num_channels as 3, as currently always transfer to RGB

    input_shape = input_shape[:2] if isinstance(input_shape, (list, tuple)) else (input_shape, input_shape)
    train_dataset = RecognitionDatset(
        train_images, train_labels, num_classes=num_classes, rescale_mode=rescale_mode, is_train=True, input_shape=input_shape, resize_method=resize_method
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, sampler=None, drop_last=False)

    test_dataset = RecognitionDatset(
        test_images, test_labels, num_classes=num_classes, rescale_mode=rescale_mode, is_train=False, input_shape=input_shape, resize_method=resize_method
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=None, drop_last=False)

    return train_dataloader, test_dataloader, total_images, num_classes, steps_per_epoch
