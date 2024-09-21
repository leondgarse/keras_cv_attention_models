import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from keras_cv_attention_models.common_layers import init_mean_std_by_rescale_mode
from keras_cv_attention_models.plot_func import draw_bboxes, show_image_with_bboxes


CV2_INTERPOLATION_MAP = {"nearest": "INTER_NEAREST", "bilinear": "INTER_LINEAR", "bicubic": "INTER_CUBIC", "area": "INTER_AREA"}


def load_from_custom_json(data_path):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)
    test_key = "validation" if "validation" in aa else "test"
    train, test, info = aa["train"], aa[test_key], aa.get("info", {})

    total_images, num_classes = len(train), info.get("num_classes", 0)
    if num_classes <= 0:
        num_classes = max([max([int(jj) for jj in ii["objects"]["label"]]) for ii in train]) + 1
        print(">>>> Using max value from train as num_classes:", num_classes)

    if "base_path" in info and len(info["base_path"]) > 0:
        base_path = os.path.expanduser(info["base_path"])
        for ii in train:
            ii["image"] = os.path.join(base_path, ii["image"])
        for ii in test:
            ii["image"] = os.path.join(base_path, ii["image"])
    return train, test, total_images, num_classes


def aspect_aware_resize_and_crop_image(image, target_shape, scale=-1, crop_y=0, crop_x=0, letterbox_pad=-1, do_pad=True, method="bilinear", antialias=False):
    import cv2

    target_shape = target_shape[:2] if isinstance(target_shape, (list, tuple)) else (target_shape, target_shape)
    letterbox_target_shape = (target_shape[0] - letterbox_pad, target_shape[1] - letterbox_pad) if letterbox_pad > 0 else target_shape
    height, width = float(image.shape[0]), float(image.shape[1])
    if scale == -1:
        scale = min(letterbox_target_shape[0] / height, letterbox_target_shape[1] / width)
    scaled_hh, scaled_ww = int(height * scale), int(width * scale)

    image = cv2.resize(image, [scaled_ww, scaled_hh], interpolation=getattr(cv2, CV2_INTERPOLATION_MAP.get(method, "INTER_LINEAR")))
    image = image[crop_y : crop_y + letterbox_target_shape[0], crop_x : crop_x + letterbox_target_shape[1]]
    cropped_shape = image.shape

    if do_pad:
        pad_top, pad_left = ((target_shape[0] - cropped_shape[0]) // 2, (target_shape[1] - cropped_shape[1]) // 2) if letterbox_pad >= 0 else (0, 0)
        image = np.pad(image, [[pad_top, target_shape[0] - scaled_hh - pad_top], [pad_left, target_shape[1] - scaled_ww - pad_left], [0, 0]])
    else:
        pad_top, pad_left = 0, 0
    return image, scale, pad_top, pad_left


def augment_hsv(image, hsv_h=0.5, hsv_s=0.5, hsv_v=0.5):
    import cv2

    gains = np.random.uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    random_color = np.arange(0, 256, dtype=gains.dtype)
    lut_hue = ((random_color * gains[0]) % 180).astype(image.dtype)
    lut_sat = np.clip(random_color * gains[1], 0, 255).astype(image.dtype)
    lut_val = np.clip(random_color * gains[2], 0, 255).astype(image.dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)


def random_perspective(
    image, bbox, label, target_shape=(640, 640), degrees=10, translate=0.1, scale=0.1, shear=10, size_thresh=2, aspect_thresh=20, area_thresh=0.1
):
    import cv2

    target_height, target_width = target_shape[:2] if isinstance(target_shape, (list, tuple)) else (target_shape, target_shape)
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # Center
    cc = np.eye(3)
    cc[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    cc[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    rr = np.eye(3)
    random_scale = random.uniform(1 - scale, 1 + scale)
    rr[:2] = cv2.getRotationMatrix2D(angle=random.uniform(-degrees, degrees), center=(0, 0), scale=random_scale)
    pre_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) * random_scale  # Used for filtering target bbox

    # Shear
    ss = np.eye(3)
    ss[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    ss[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    tt = np.eye(3)
    tt[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * target_width  # x translation (pixels)
    tt[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * target_height  # y translation (pixels)

    # Combined rotation matrix
    mm = tt @ ss @ rr @ cc  # order of operations (right to left) is IMPORTANT
    image = cv2.warpAffine(image, mm[:2], dsize=(target_width, target_height), borderValue=(114, 114, 114))

    # warp points
    bbox = bbox[:, [1, 0, 3, 2, 3, 0, 1, 2]].reshape(-1, 2)  # x1y1, x2y2, x1y2, x2y1
    bbox = np.pad(bbox, [[0, 0], [0, 1]], constant_values=1)
    bbox = bbox @ mm.T  # transform
    bbox = bbox[:, :2].reshape(-1, 8)

    # create new boxes
    xx, yy = bbox[:, [0, 2, 4, 6]], bbox[:, [1, 3, 5, 7]]
    bbox = np.concatenate((yy.min(1), xx.min(1), yy.max(1), xx.max(1))).reshape(4, -1).T

    # clip boxes
    bbox[:, [0, 2]] = bbox[:, [0, 2]].clip(0, target_height)
    bbox[:, [1, 3]] = bbox[:, [1, 3]].clip(0, target_width)

    # filter candidates
    height, width = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    valid_height, valid_width = height > size_thresh, width > size_thresh
    valid_area = height * width / (pre_area + 1e-16) > area_thresh
    valid_height_aspect, valid_width_aspect = height / (width + 1e-16) < aspect_thresh, width / (height + 1e-16) < aspect_thresh
    valid_candidates = valid_height & valid_width & valid_area & valid_height_aspect & valid_width_aspect
    return image, bbox[valid_candidates], label[valid_candidates]


def combine_mosaic(images, bboxes, labels, target_shape=(640, 640)):
    # loads images in a mosaic, bboxes: [top, left, bottom, right]
    target_height, target_width = target_shape[:2] if isinstance(target_shape, (list, tuple)) else (target_shape, target_shape)
    hh_center = int(random.uniform(target_height // 2, 2 * target_height - target_height // 2))
    ww_center = int(random.uniform(target_width // 2, 2 * target_width - target_width // 2))

    paste_height, paste_width = target_height * 2, target_width * 2
    mosaic_image = np.full((paste_height, paste_width, 3), 114, dtype=np.uint8)  # base image with 4 tiles
    mosaic_bboxes, mosaic_labels = [], []

    for order in range(4):
        cur_image = images[order]
        height, width = cur_image.shape[0], cur_image.shape[1]
        if order == 0:
            paste_top, paste_left, paste_bottom, paste_right = max(hh_center - height, 0), max(ww_center - width, 0), hh_center, ww_center
            cut_top, cut_left, cut_bottom, cut_right = height - (hh_center - paste_top), width - (ww_center - paste_left), height, width
        elif order == 1:
            paste_top, paste_left, paste_bottom, paste_right = max(hh_center - height, 0), ww_center, hh_center, min(ww_center + width, paste_width)
            cut_top, cut_left, cut_bottom, cut_right = height - (hh_center - paste_top), 0, height, min(paste_right - ww_center, width)
        elif order == 2:
            paste_top, paste_left, paste_bottom, paste_right = hh_center, max(ww_center - width, 0), min(hh_center + height, paste_height), ww_center
            cut_top, cut_left, cut_bottom, cut_right = 0, width - (ww_center - paste_left), min(paste_bottom - hh_center, height), width
        elif order == 3:
            paste_top, paste_left, paste_bottom, paste_right = hh_center, ww_center, min(hh_center + height, paste_height), min(ww_center + width, paste_width)
            cut_top, cut_left, cut_bottom, cut_right = 0, 0, min(paste_bottom - hh_center, height), min(paste_right - ww_center, width)

        mosaic_image[paste_top:paste_bottom, paste_left:paste_right] = cur_image[cut_top:cut_bottom, cut_left:cut_right]
        hh_offset, ww_offset = paste_top - cut_top, paste_left - cut_left
        mosaic_bboxes.append(bboxes[order] * [height, width, height, width] + [hh_offset, ww_offset, hh_offset, ww_offset])
        mosaic_labels.append(labels[order])
    mosaic_bboxes = np.clip(np.concatenate(mosaic_bboxes, axis=0), 0, [paste_height, paste_width, paste_height, paste_width])
    mosaic_labels = np.concatenate(mosaic_labels, axis=0)
    return mosaic_image, mosaic_bboxes, mosaic_labels


class DetectionDataset(Dataset):  # for training/testing
    """
    >>> from keras_cv_attention_models.coco import torch_data
    >>> from keras_cv_attention_models.plot_func import show_image_with_bboxes
    >>> train, test = torch_data.load_from_custom_json('datasets/coco_dog_cat/detections.json')[:2]
    >>> aa = torch_data.DetectionDataset(train)
    >>> image, bbox, label = aa.__getitem__(0)
    >>> ax = show_image_with_bboxes(image.numpy().transpose([1, 2, 0]), bbox, label, indices_2_labels={0: 'cat', 1: 'dog'})
    >>> ax.get_figure().savefig('aa.jpg')
    """

    def __init__(self, data, image_size=(640, 640), batch_size=16, rescale_mode="raw01", is_train=True, mosaic=1.0, max_mosaic_cache_len=1024):
        self.data, self.batch_size, self.rescale_mode, self.is_train, self.mosaic = data, batch_size, rescale_mode, is_train, mosaic
        self.target_shape = image_size[:2] if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.mean, self.std = init_mean_std_by_rescale_mode(rescale_mode)

        # self.rect = False if is_train else rect
        # import ultralytics
        # cfg = ultralytics.cfg.get_cfg(ultralytics.utils.DEFAULT_CFG)
        # cfg.degrees, cfg.translate, cfg.scale, cfg.shear, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v
        self.degrees, self.translate, self.scale, self.shear = 0.0, 0.1, 0.5, 0.0
        self.hsv_h, self.hsv_s, self.hsv_v = 0.015, 0.7, 0.4
        self.fliplr = 0.5

        """ Convert items all to ndarray """
        for datapoint in self.data:
            datapoint["objects"]["bbox"] = np.array(datapoint["objects"]["bbox"], dtype="float32").reshape([-1, 4])
            datapoint["objects"]["label"] = np.array(datapoint["objects"]["label"], dtype="int64")

        import cv2

        self.imread = cv2.imread

        """ Cache first for using in mosaic mix """
        self.cached_images, self.cached_image_indexes, self.cache_length = [None] * len(data), [], min(batch_size * 32, max_mosaic_cache_len)
        if is_train:
            for index in np.random.choice(range(len(data)), 4, replace=False):
                self.cached_image_indexes.append(index)
                self.cached_images[index] = self.__imread__(data[index]["image"])

    def __len__(self):
        return len(self.data)

    def __imread__(self, image_path):
        image = self.imread(image_path)
        return aspect_aware_resize_and_crop_image(image, target_shape=self.target_shape, do_pad=False, method="bilinear", antialias=False)[0]

    def __process_eval__(self, index, image_path, bbox, label):
        image = self.__imread__(image_path)
        bbox *= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]
        image = np.pad(image, [[0, self.target_shape[0] - image.shape[0]], [0, self.target_shape[1] - image.shape[1]], [0, 0]])
        bbox /= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]
        return image, bbox, label

    def __process_train__(self, index, image_path, bbox, label):
        """Cache read images"""
        if index in self.cached_image_indexes:  # Seldom should this happen
            image = self.cached_images[index]
        else:
            image = self.__imread__(image_path)
            if len(self.cached_image_indexes) == self.cache_length:
                clear_index = self.cached_image_indexes[0]
                self.cached_image_indexes = self.cached_image_indexes[1:]
                self.cached_images[clear_index] = None
            # print(f">>>> {len(self.cached_image_indexes) = } {len([ii for ii in self.cached_images if ii is not None])}")
            self.cached_image_indexes.append(index)
            self.cached_images[index] = image

        """ Mosaic mix """
        if random.random() < self.mosaic:
            images, bboxes, labels = [image], [bbox], [label]
            for index in np.random.choice(self.cached_image_indexes, 3, replace=False):  # 3 additional image indices from cached ones
                datapoint = self.data[index]
                images.append(self.cached_images[index])
                bboxes.append(datapoint["objects"]["bbox"])
                labels.append(datapoint["objects"]["label"])
            image, bbox, label = combine_mosaic(images, bboxes, labels, target_shape=self.target_shape)
        else:
            bbox *= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]

        image, bbox, label = random_perspective(
            image, bbox, label, target_shape=self.target_shape, degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear
        )  # Also process image as target_shape
        image = augment_hsv(image, hsv_h=self.hsv_h, hsv_s=self.hsv_s, hsv_v=self.hsv_v)
        bbox /= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]  # normalized to [0, 1]

        if random.random() < self.fliplr:
            image = np.fliplr(image)
            bbox[:, [1, 3]] = 1 - bbox[:, [3, 1]]
        return image, bbox, label

    def __getitem__(self, index):
        datapoint = self.data[index]
        image_path, objects = datapoint["image"], datapoint["objects"]
        bbox, label = objects["bbox"], objects["label"]
        image, bbox, label = self.__process_train__(index, image_path, bbox, label) if self.is_train else self.__process_eval__(index, image_path, bbox, label)

        image = image.transpose(2, 0, 1)[::-1]  # BGR -> channels first -> RGB
        image = (torch.from_numpy(np.ascontiguousarray(image)).float() - self.mean) / self.std
        return image, torch.from_numpy(bbox).float(), torch.from_numpy(label).float()


def collate_wrapper(batch):
    images, bboxes, labels = list(zip(*batch))
    batch_ids = [torch.as_tensor([id] * len(ii)) for id, ii in enumerate(labels)]
    batch_ids, bboxes, labels = torch.concat(batch_ids, dim=0), torch.concat(bboxes, dim=0), torch.concat(labels, dim=0)
    # print(f">>>> {batch_ids.shape = }, {bboxes.shape = }, {labels.shape = }")
    return torch.stack(images), torch.concat([batch_ids[:, None], bboxes, labels[:, None]], dim=-1)


def init_dataset(data_path, batch_size=64, image_size=(640, 640), rescale_mode="raw01", num_workers=8, max_mosaic_cache_len=1024, with_info=False):
    """
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> from keras_cv_attention_models.coco import torch_data
    >>> from keras_cv_attention_models.plot_func import show_image_with_bboxes
    >>> train, test = torch_data.init_dataset('datasets/coco_dog_cat/detections.json')
    >>> images, labels = next(iter(train))
    >>> image, label = images[0].numpy().transpose([1, 2, 0]), labels[labels[:, 0] == 0].numpy()
    >>> ax = show_image_with_bboxes(image, label[:, 1:-1], label[:, -1], indices_2_labels={0: 'cat', 1: 'dog'})
    >>> ax.get_figure().savefig('aa.jpg')
    """
    train, test, total_images, num_classes = load_from_custom_json(data_path)

    train_dataset = DetectionDataset(train, rescale_mode=rescale_mode, is_train=True, image_size=image_size, max_mosaic_cache_len=max_mosaic_cache_len)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=False
    )

    test_dataset = DetectionDataset(test, rescale_mode=rescale_mode, is_train=False, image_size=image_size, max_mosaic_cache_len=max_mosaic_cache_len)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=False
    )

    return (train_dataloader, test_dataloader, total_images, num_classes) if with_info else (train_dataloader, test_dataloader)
