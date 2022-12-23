import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.imagenet.data import init_mean_std_by_rescale_mode, tf_imread, random_crop_and_resize_image
from keras_cv_attention_models.coco import anchors_func

COCO_LABELS = """person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign,
    parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie,
    suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
    bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
    cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
    toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""
COCO_80_LABEL_DICT = {id: ii.strip() for id, ii in enumerate(COCO_LABELS.split(","))}
INVALID_ID_90 = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]
COCO_90_LABEL_DICT = {id: ii for id, ii in zip(set(range(90)) - set(INVALID_ID_90), COCO_80_LABEL_DICT.values())}
COCO_90_LABEL_DICT.update({ii: "Unknown" for ii in INVALID_ID_90})
COCO_80_to_90_LABEL_DICT = {id_80: id_90 for id_80, id_90 in enumerate(set(range(90)) - set(INVALID_ID_90))}


""" Bboxes augment """


def rerange_scale_offset_to_01(source_height, source_width, target_height, target_width, scale_hh, scale_ww, offset_hh=0, offset_ww=0):
    # Input: image size firstly rescale with (scale_hh, scale_ww), then crop as [offset_hh: offset_hh + target_height, offset_ww: offset_ww + target_width]
    # Output: coords or bbox value in (0, 1) -> * scale - offset
    source_height, source_width = tf.cast(source_height, "float32"), tf.cast(source_width, "float32")
    target_height, target_width = tf.cast(target_height, "float32"), tf.cast(target_width, "float32")
    scale_hh_01, scale_ww_01 = source_height * scale_hh / target_height, source_width * scale_ww / target_width
    crop_hh_01, crop_ww_01 = tf.cast(offset_hh, "float32") / target_height, tf.cast(offset_ww, "float32") / target_width
    return scale_hh_01, scale_ww_01, crop_hh_01, crop_ww_01


def inverse_affine_matrix_single_6(affine):
    scale_inverse = affine[0] * affine[4] - affine[3] * affine[1]
    affine_aa = [affine[4], -affine[1], affine[5] * affine[1] - affine[2] * affine[4]]
    affine_bb = [-affine[3], affine[0], affine[2] * affine[3] - affine[5] * affine[0]]
    return tf.convert_to_tensor(affine_aa + affine_bb, dtype=affine.dtype) / scale_inverse


def bboxes_apply_affine(bboxes, affine, input_shape, inverse=True):
    height, width = tf.cast(input_shape[0], bboxes.dtype), tf.cast(input_shape[1], bboxes.dtype)
    top, left, bottom, right = bboxes[:, 0] * height, bboxes[:, 1] * width, bboxes[:, 2] * height, bboxes[:, 3] * width
    affine = tf.squeeze(affine)[:6]
    if inverse:
        affine = inverse_affine_matrix_single_6(affine)
    affine = tf.reshape(affine, [2, 3])

    # 4 corners points: [left, top, right, bottom, left, bottom, right, top]
    corners = tf.gather(bboxes * [height, width, height, width], [1, 0, 3, 2, 1, 2, 3, 0], axis=-1)
    # -> [[left, top], [right, bottom], [left, bottom], [right, top]]
    corners = tf.reshape(corners, [-1, 4, 2])
    # pad 1
    corners = tf.concat([corners, tf.ones_like(corners[:, :, :1])], axis=-1)
    # apply affine transform
    corners_transformed = tf.matmul(corners, affine, transpose_b=True)

    new_left = tf.minimum(corners_transformed[:, 0, 0], corners_transformed[:, 2, 0]) / width
    new_top = tf.minimum(corners_transformed[:, 0, 1], corners_transformed[:, 3, 1]) / height
    new_right = tf.maximum(corners_transformed[:, 1, 0], corners_transformed[:, 3, 0]) / width
    new_bottom = tf.maximum(corners_transformed[:, 1, 1], corners_transformed[:, 2, 1]) / height
    return tf.stack([new_top, new_left, new_bottom, new_right], axis=-1)


def refine_bboxes_labels_single(bboxes, labels):
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    picking_indices = tf.where(tf.not_equal((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]), 0))
    bboxes, labels = tf.gather_nd(bboxes, picking_indices), tf.gather_nd(labels, picking_indices)
    return bboxes, labels


""" Image Augment """


def random_largest_crop_and_resize_images(images, target_shape, method="bilinear", antialias=False):
    if tf.rank(images) == 3:
        height, width = tf.cast(tf.shape(images)[0], "float32"), tf.cast(tf.shape(images)[1], "float32")
    else:
        height, width = tf.cast(tf.shape(images)[1], "float32"), tf.cast(tf.shape(images)[2], "float32")
    target_height, target_width = tf.cast(target_shape[0], "float32"), tf.cast(target_shape[1], "float32")
    scale = tf.maximum(target_height / height, target_width / width)
    scaled_hh, scaled_ww = tf.cast(tf.math.ceil(height * scale), "int32"), tf.cast(tf.math.ceil(width * scale), "int32")
    images = tf.image.resize(images, [scaled_hh, scaled_ww], method=method, antialias=antialias)

    # print(target_shape)
    crop_hh = tf.cond(scaled_hh > target_shape[0], lambda: tf.random.uniform((), 0, scaled_hh - target_shape[0], dtype="int32"), lambda: 0)
    crop_ww = tf.cond(scaled_ww > target_shape[1], lambda: tf.random.uniform((), 0, scaled_ww - target_shape[1], dtype="int32"), lambda: 0)
    # crop_hh = tf.random.uniform((), 0, tf.maximum(scaled_hh - target_shape[0], 1), dtype='int32')
    # crop_ww = tf.random.uniform((), 0, tf.maximum(scaled_ww - target_shape[1], 1), dtype='int32')
    if tf.rank(images) == 3:
        images = images[crop_hh : crop_hh + target_shape[0], crop_ww : crop_ww + target_shape[1]]
    else:
        images = images[:, crop_hh : crop_hh + target_shape[0], crop_ww : crop_ww + target_shape[1]]
    return images, scale, crop_hh, crop_ww


def get_image_aspect_aware_random_scale_crop(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
    """https://github.com/google/automl/tree/master/efficientdet/dataloader.py#L67"""
    random_scale_factor = tf.random.uniform((), scale_min, scale_max)
    scaled_y, scaled_x = random_scale_factor * target_shape[0], random_scale_factor * target_shape[1]
    height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
    random_image_scale = tf.minimum(scaled_y / height, scaled_x / width)

    # Select non-zero random offset (x, y) if scaled image is larger than self._output_size.
    height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
    scaled_height, scaled_width = height * random_image_scale, width * random_image_scale
    offset_y, offset_x = tf.maximum(0.0, scaled_height - target_shape[0]), tf.maximum(0.0, scaled_width - target_shape[1])
    random_offset_y, random_offset_x = offset_y * tf.random.uniform([], 0, 1), offset_x * tf.random.uniform([], 0, 1)
    random_offset_y, random_offset_x = tf.cast(random_offset_y, tf.int32), tf.cast(random_offset_x, tf.int32)
    return random_image_scale, random_offset_y, random_offset_x


def aspect_aware_resize_and_crop_image(image, target_shape, scale=-1, crop_y=0, crop_x=0, letterbox_pad=-1, method="bilinear", antialias=False):
    letterbox_target_shape = (target_shape[0] - letterbox_pad, target_shape[1] - letterbox_pad) if letterbox_pad > 0 else target_shape
    height, width = tf.cast(tf.shape(image)[0], "float32"), tf.cast(tf.shape(image)[1], "float32")
    if scale == -1:
        scale = tf.minimum(letterbox_target_shape[0] / height, letterbox_target_shape[1] / width)
    scaled_hh, scaled_ww = int(height * scale), int(width * scale)
    image = tf.image.resize(image, [scaled_hh, scaled_ww], method=method, antialias=antialias)
    image = image[crop_y : crop_y + letterbox_target_shape[0], crop_x : crop_x + letterbox_target_shape[1]]
    cropped_shape = tf.shape(image)

    pad_top, pad_left = ((target_shape[0] - cropped_shape[0]) // 2, (target_shape[1] - cropped_shape[1]) // 2) if letterbox_pad >= 0 else (0, 0)
    image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, target_shape[0], target_shape[1])
    return image, scale, pad_top, pad_left


def random_flip_left_right_with_bboxes(image, bboxes, probability=0.5):
    # For box, left = 1 - right, right = 1 - left
    return tf.cond(
        tf.random.uniform(()) < probability,
        lambda: (tf.image.flip_left_right(image), tf.gather(bboxes, [0, 3, 2, 1], axis=1) * [1, -1, 1, -1] + [0, 1, 0, 1]),
        lambda: (image, bboxes),
    )


def random_hsv(image, hue_delta=0.015, saturation_delta=0.7, brightness_delta=0.4, contrast_delta=0, show_sample=False):
    # augment_hsv https://github.com/WongKinYiu/yolor/blob/main/utils/datasets.py#L941
    if show_sample:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        aa = tf.concat([tf.image.adjust_hue(image, -hue_delta), tf.image.adjust_hue(image, hue_delta)], axis=1)
        bb = tf.concat([tf.image.adjust_saturation(image, 1 - saturation_delta), tf.image.adjust_saturation(image, 1 + saturation_delta)], axis=1)
        cc = tf.concat([tf.image.adjust_brightness(image, -brightness_delta), tf.image.adjust_brightness(image, brightness_delta)], axis=1)
        dd = tf.concat([tf.image.adjust_contrast(image, 1 - contrast_delta), tf.image.adjust_contrast(image, 1 + contrast_delta)], axis=1)
        plt.imshow(tf.concat([aa, bb, cc, dd], axis=0))
        plt.axis("off")
        plt.tight_layout()

    image = tf.image.random_hue(image, hue_delta)
    image = tf.image.random_saturation(image, 1 - saturation_delta, 1 + saturation_delta)
    image = tf.image.random_brightness(image, brightness_delta)
    if contrast_delta > 0:
        image = tf.image.random_contrast(image, 1 - contrast_delta, 1 + contrast_delta)
    return image


""" Mosaic mix """


def mosaic_mix_batch(images, bboxes, labels, split_center_min=0.25, split_center_max=0.75):
    batch_size = tf.shape(images)[0]
    _, hh, ww, _ = images.shape
    split_hh = tf.cast(tf.random.uniform((), split_center_min * hh, split_center_max * hh), "int32")
    split_ww = tf.cast(tf.random.uniform((), split_center_min * ww, split_center_max * ww), "int32")
    # print(split_hh, split_ww)

    # top_left, top_right, bottom_left, bottom_right
    # sub_hh_wws = [[split_hh, split_ww], [split_hh, ww - split_ww], [hh - split_hh, split_ww], [hh - split_hh, ww - split_ww]]
    starts = [[0, 0], [0, split_ww], [split_hh, 0], [split_hh, split_ww]]
    ends = [[split_hh, split_ww], [split_hh, ww], [hh, split_ww], [hh, ww]]
    mixed_images, mixed_bboxes, mixed_labels = [], [], []
    for (top, left), (bottom, right) in zip(starts, ends):
        sub_hh, sub_ww = bottom - top, right - left
        pick_indices = tf.random.shuffle(tf.range(batch_size))
        cur_images = tf.gather(images, pick_indices)
        cur_images, scale, crop_hh, crop_ww = random_largest_crop_and_resize_images(cur_images, [sub_hh, sub_ww])
        mixed_images.append(cur_images)
        # print(f"{cur_images.shape = }, {scale = }")

        cur_bboxes, cur_labels = tf.gather(bboxes, pick_indices), tf.gather(labels, pick_indices)
        target_height, target_width = 1, 1  # Don't scale back to [0, 1]
        scale_hh, scale_ww, crop_hh, crop_ww = rerange_scale_offset_to_01(hh, ww, target_height, target_width, scale, scale, crop_hh, crop_ww)
        cur_bboxes = cur_bboxes * [scale_hh, scale_ww, scale_hh, scale_ww] - [crop_hh, crop_ww, crop_hh, crop_ww]

        """ Re-fine batch bboxes """
        sub_hh, sub_ww = tf.cast(sub_hh, cur_bboxes.dtype), tf.cast(sub_ww, cur_bboxes.dtype)
        cur_bboxes = tf.clip_by_value(cur_bboxes, 0, [sub_hh, sub_ww, sub_hh, sub_ww])
        mask_cond = tf.not_equal((cur_bboxes[:, :, 2] - cur_bboxes[:, :, 0]) * (cur_bboxes[:, :, 3] - cur_bboxes[:, :, 1]), 0)
        cur_bboxes += [top, left, top, left]
        cur_bboxes = cur_bboxes * tf.expand_dims(tf.cast(mask_cond, cur_bboxes.dtype), -1)
        cur_labels = tf.where(mask_cond, cur_labels, tf.zeros_like(cur_labels))

        mixed_bboxes.append(cur_bboxes)
        mixed_labels.append(cur_labels)

    top_images = tf.concat([mixed_images[0], mixed_images[1]], axis=2)
    bottom_images = tf.concat([mixed_images[2], mixed_images[3]], axis=2)
    mixed_images = tf.concat([top_images, bottom_images], axis=1)
    mixed_images.set_shape([None, hh, ww, 3])
    mixed_bboxes = tf.concat(mixed_bboxes, axis=1) / [hh, ww, hh, ww]
    mixed_labels = tf.concat(mixed_labels, axis=1)
    # print(f"{top_images.shape = }, {bottom_images.shape = }, {mix.shape = }, {mixed_images.shape = }, {mixed_labels.shape = }")
    return mixed_images, (mixed_bboxes, mixed_labels)


""" Dataset """


class RandomProcessImageWithBboxes:
    def __init__(
        self,
        target_shape=(300, 300),
        max_labels_per_image=100,
        random_crop_mode=0,  # 0 for eval mode, (0, 1) for random crop, 1 for random largest crop, > 1 for random scale
        resize_method="bilinear",
        resize_antialias=False,
        color_augment_method="random_hsv",
        magnitude=0,
        num_layers=2,
        use_color_increasing=True,
        **randaug_kwargs,
    ):
        self.max_labels_per_image = max_labels_per_image
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.resize_method, self.resize_antialias, self.random_crop_mode, self.magnitude = resize_method, resize_antialias, random_crop_mode, magnitude
        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            if color_augment_method is None:
                self.randaug_wo_pos = lambda image: image
            elif isinstance(color_augment_method, str) and color_augment_method.lower() == "autoaug":
                self.randaug_wo_pos = augment.AutoAugment(augmentation_name="simple")
            elif isinstance(color_augment_method, str) and color_augment_method.lower() == "randaug":
                # TODO: Need to pick color related methods, "Invert" / "Posterize" may not working well here.
                print(">>>> RandAugment: magnitude = %d" % magnitude)
                self.randaug_wo_pos = augment.RandAugment(
                    num_layers=num_layers,
                    magnitude=magnitude,
                    use_cutout=False,
                    use_color_increasing=use_color_increasing,
                    use_positional_related_ops=False,  # Set False to exlude [shear, rotate, translate]
                    **randaug_kwargs,
                )
            elif isinstance(color_augment_method, str):
                self.randaug_wo_pos = random_hsv
            else:
                print(">>>> Using user defined color_augment_method: {}".format(getattr(color_augment_method, "__name__", color_augment_method)))
                self.randaug_wo_pos = color_augment_method

            # RandAugment positional related ops. Including [shear, rotate, translate], Also returns affine transform matrix
            # self.pos_randaug = augment.PositionalRandAugment(num_layers=num_layers, magnitude=magnitude, **randaug_kwargs)

    def __random_crop_and_resize__(self, image, bbox):
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        if self.random_crop_mode > 1:
            scale, crop_hh, crop_ww = get_image_aspect_aware_random_scale_crop((height, width), self.target_shape, scale_max=self.random_crop_mode)
            image, scale, _, _ = aspect_aware_resize_and_crop_image(
                image, self.target_shape, scale, crop_hh, crop_ww, method=self.resize_method, antialias=self.resize_antialias
            )
            scale_hh, scale_ww, crop_hh, crop_ww = rerange_scale_offset_to_01(
                height, width, self.target_shape[0], self.target_shape[1], scale, scale, crop_hh, crop_ww
            )
        elif self.random_crop_mode == 1:
            image, scale, crop_hh, crop_ww = random_largest_crop_and_resize_images(image, self.target_shape, self.resize_method, self.resize_antialias)
            scale_hh, scale_ww, crop_hh, crop_ww = rerange_scale_offset_to_01(
                height, width, self.target_shape[0], self.target_shape[1], scale, scale, crop_hh, crop_ww
            )
        elif self.random_crop_mode > 0 and self.random_crop_mode < 1:
            image, scale_hh, scale_ww, crop_hh, crop_ww = random_crop_and_resize_image(
                image, self.target_shape, scale=(self.random_crop_mode, 1.0), method=self.resize_method, antialias=self.resize_antialias
            )
        else:
            image, scale, _, _ = aspect_aware_resize_and_crop_image(image, self.target_shape, method=self.resize_method, antialias=self.resize_antialias)
            scale_hh, scale_ww, crop_hh, crop_ww = rerange_scale_offset_to_01(height, width, self.target_shape[0], self.target_shape[1], scale, scale)

        bbox = bbox * [scale_hh, scale_ww, scale_hh, scale_ww] - [crop_hh, crop_ww, crop_hh, crop_ww]
        return image, bbox

    def __call__(self, datapoint):
        image = datapoint["image"]
        objects = datapoint["objects"]
        bbox, label, is_crowd = tf.cast(objects["bbox"], tf.float32), objects["label"], objects.get("is_crowd", None)
        if len(image.shape) < 2:
            image = tf_imread(image)
        if is_crowd is not None:
            is_not_crowd = is_crowd == False
            bbox, label = tf.boolean_mask(bbox, is_not_crowd), tf.boolean_mask(label, is_not_crowd)

        if self.magnitude >= 0:
            image, bbox = random_flip_left_right_with_bboxes(image, bbox)

        image, bbox = self.__random_crop_and_resize__(image, bbox)
        bbox, label = refine_bboxes_labels_single(bbox, label)

        if self.magnitude > 0:
            image.set_shape([*self.target_shape[:2], 3])
            image = self.randaug_wo_pos(image)

        should_pad = self.max_labels_per_image - tf.shape(bbox)[0]
        # label starts from 0 -> person, add 1 here, differs from padded values
        bbox, label = tf.cond(
            should_pad > 0,
            lambda: (tf.pad(bbox, [[0, should_pad], [0, 0]]), tf.pad(label + 1, [[0, should_pad]])),
            lambda: (bbox[: self.max_labels_per_image], label[: self.max_labels_per_image] + 1),
        )

        image = tf.cast(image, tf.float32)
        image.set_shape([*self.target_shape[:2], 3])
        return image, (bbox, label)


class PositionalRandAugmentWithBboxes:
    def __init__(self, magnitude=0, num_layers=2, max_labels_per_image=100, positional_augment_methods="rts", **randaug_kwargs):
        from keras_cv_attention_models.imagenet import augment

        self.pos_randaug = augment.PositionalRandAugment(
            num_layers=num_layers, magnitude=magnitude, positional_augment_methods=positional_augment_methods, **randaug_kwargs
        )
        self.max_labels_per_image = max_labels_per_image

    def __call_single__(self, inputs):
        image, bbox, label = inputs
        image, affine_matrix = self.pos_randaug(image)
        pick = tf.where(label != 0)
        bbox, label = tf.gather_nd(bbox, pick), tf.gather_nd(label, pick)
        bbox = bboxes_apply_affine(bbox, affine_matrix, input_shape=image.shape)
        bbox, label = refine_bboxes_labels_single(bbox, label)

        should_pad = self.max_labels_per_image - tf.shape(bbox)[0]
        bbox, label = tf.cond(
            should_pad > 0,
            lambda: (tf.pad(bbox, [[0, should_pad], [0, 0]]), tf.pad(label, [[0, should_pad]])),
            lambda: (bbox[: self.max_labels_per_image], label[: self.max_labels_per_image]),
        )
        return image, bbox, label

    def __call__(self, xx, yy):
        image, bbox, label = tf.map_fn(self.__call_single__, (xx, yy[0], yy[1]))
        return image, (bbox, label)


def to_one_hot_with_class_mark(anchor_bboxes_with_label, num_classes=80):
    # dest_boxes, anchor_classes = anchor_bboxes_with_label[:, :4], anchor_bboxes_with_label[:, -1]
    dest_boxes, anchor_classes = tf.split(anchor_bboxes_with_label, [4, 1], axis=-1)
    one_hot_labels = tf.one_hot(tf.cast(anchor_classes[..., 0], "int32") - 1, num_classes)  # [1, 81] -> [0, 80]
    # Mark iou < ignore_threshold as 0, ignore_threshold < iou < overlap_threshold as -1, iou > overlap_threshold as 1
    marks = tf.where(anchor_classes > 0, tf.ones_like(anchor_classes), anchor_classes)
    # marks = tf.expand_dims(marks, -1)
    one_hot_labels, marks = tf.cast(one_hot_labels, dest_boxes.dtype), tf.cast(marks, dest_boxes.dtype)
    return tf.concat([dest_boxes, one_hot_labels, marks], axis=-1)


def __bboxes_labels_batch_func__(bboxes, labels, anchors, empty_label, num_classes=80):
    bbox_labels = tf.concat([bboxes, tf.cast(tf.expand_dims(labels, -1), bboxes.dtype)], axis=-1)
    bbox_process = lambda xx: tf.cond(
        tf.reduce_any(xx[:, -1] > 0),  # If contains any valid bbox and label
        lambda: to_one_hot_with_class_mark(anchors_func.assign_anchor_classes_by_iou_with_bboxes(xx, anchors), num_classes),
        lambda: empty_label,
    )
    return tf.map_fn(bbox_process, bbox_labels)


def __yolor_bboxes_labels_batch_func__(bboxes, labels, anchor_ratios, feature_sizes, empty_label, num_classes=80):
    bbox_labels = tf.concat([bboxes, tf.cast(tf.expand_dims(labels, -1), bboxes.dtype)], axis=-1)
    bbox_process = lambda xx: tf.cond(
        tf.reduce_any(xx[:, -1] > 0),  # If contains any valid bbox and label
        lambda: to_one_hot_with_class_mark(anchors_func.yolor_assign_anchors(xx, anchor_ratios, feature_sizes), num_classes),
        lambda: empty_label,
    )
    return tf.map_fn(bbox_process, bbox_labels)


def detection_dataset_from_custom_json(data_path, with_info=False):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)

    test_key = "validation" if "validation" in aa else "test"
    train, test, info = aa["train"], aa[test_key], aa["info"]
    total_images, num_classes = len(train), info["num_classes"]
    objects_signature = {"bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32), "label": tf.TensorSpec(shape=(None,), dtype=tf.int64)}
    output_signature = {"image": tf.TensorSpec(shape=(), dtype=tf.string), "objects": objects_signature}
    train_ds = tf.data.Dataset.from_generator(lambda: (ii for ii in train), output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(lambda: (ii for ii in test), output_signature=output_signature)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(train))).with_options(options)
    test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(len(test))).with_options(options)
    dataset = {"train": train_ds, test_key: test_ds}
    return (dataset, total_images, num_classes) if with_info else dataset


def init_dataset(
    data_name="coco/2017",  # dataset params
    input_shape=(256, 256),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    max_labels_per_image=100,
    anchors_mode="efficientdet",
    anchor_pyramid_levels=[3, 7],
    anchor_scale=4,  # For efficientdet anchors only
    aspect_ratios=(1, 2, 0.5),  # For efficientdet anchors only
    num_scales=3,  # For efficientdet anchors only
    rescale_mode="torch",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    resize_antialias=False,
    random_crop_mode=1.0,  # [augment] 0 for eval mode, (0, 1) for random crop, 1 for random largest crop, > 1 for random scale
    mosaic_mix_prob=0.0,  # [augment] 0 for disable, > 0 for enable
    color_augment_method="random_hsv",  # [augment] one of ["random_hsv", "autoaug", "randaug"], or totally custom one like `lambda image: image`
    positional_augment_methods="rts",  # [augment] Positional augment method besides scale, combine of r: rotate, t: transplate, s: shear, x: scale_x + scale_y
    magnitude=0,
    num_layers=2,
    seed=None,
    **augment_kwargs,  # Too many...
):
    import tensorflow_datasets as tfds

    is_tpu = True if len(tf.config.list_logical_devices("TPU")) > 0 else False  # Set True for try_gcs and drop_remainder

    if data_name.endswith(".json"):
        dataset, total_images, num_classes = detection_dataset_from_custom_json(data_name, with_info=True)
    else:
        dataset, info = tfds.load(data_name, with_info=True, try_gcs=is_tpu)
        num_classes = info.features["objects"]["label"].num_classes
        total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE

    train_process = RandomProcessImageWithBboxes(
        target_shape=input_shape,
        max_labels_per_image=max_labels_per_image,
        random_crop_mode=random_crop_mode,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        color_augment_method=color_augment_method,
        magnitude=magnitude,
        num_layers=num_layers,
        **augment_kwargs,
    )
    train_dataset = dataset["train"].shuffle(buffer_size, seed=seed).map(train_process).batch(batch_size, drop_remainder=is_tpu)
    # return train_dataset

    if mosaic_mix_prob > 0:
        mosaic_mix = lambda xx, yy: tf.cond(
            tf.random.uniform(()) > mosaic_mix_prob,
            lambda: (xx, yy),
            lambda: mosaic_mix_batch(xx, yy[0], yy[1]),
        )
        train_dataset = train_dataset.map(mosaic_mix, num_parallel_calls=AUTOTUNE)
    # return train_dataset

    if magnitude > 0 and positional_augment_methods is not None and len(positional_augment_methods) != 0:
        # Apply randaug rotate / shear / transform after mosaic mix
        max_labels_per_image = (max_labels_per_image * 4) if mosaic_mix_prob > 0 else max_labels_per_image
        pos_aug = PositionalRandAugmentWithBboxes(magnitude, num_layers, max_labels_per_image, positional_augment_methods, **augment_kwargs)
        print(">>>> positional augment methods:", pos_aug.pos_randaug.available_ops)
        train_dataset = train_dataset.map(pos_aug, num_parallel_calls=AUTOTUNE)

    if anchors_mode == anchors_func.ANCHOR_FREE_MODE:  # == "anchor_free"
        # Don't need anchors here, anchor assigning is after getting model predictions.
        bbox_process = lambda bb: to_one_hot_with_class_mark(tf.concat([bb[0], tf.cast(tf.expand_dims(bb[1], -1), bb[0].dtype)], axis=-1), num_classes)
    elif anchors_mode == anchors_func.YOLOR_MODE:  # == "yolor":
        anchor_ratios, feature_sizes = anchors_func.get_yolor_anchors(input_shape[:2], anchor_pyramid_levels, is_for_training=True)
        total_anchors = tf.cast(anchor_ratios.shape[1] * tf.reduce_sum(feature_sizes[:, 0] * feature_sizes[:, 1]), tf.int32)
        empty_label = tf.zeros([total_anchors, 4 + num_classes + 1])  # All 0
        bbox_process = lambda bb: __yolor_bboxes_labels_batch_func__(bb[0], bb[1], anchor_ratios, feature_sizes, empty_label, num_classes)
    else:
        # grid_zero_start = True if anchor_grid_zero_start == "auto" else anchor_grid_zero_start
        grid_zero_start = False  # Use this till meet some others new
        anchors = anchors_func.get_anchors(input_shape[:2], anchor_pyramid_levels, aspect_ratios, num_scales, anchor_scale, grid_zero_start)
        num_anchors = anchors.shape[0]
        empty_label = tf.zeros([num_anchors, 4 + num_classes + 1])  # All 0
        bbox_process = lambda bb: __bboxes_labels_batch_func__(bb[0], bb[1], anchors, empty_label, num_classes)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std
    train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), bbox_process(yy)), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # return train_dataset

    """ Test dataset """
    test_dataset = dataset.get("validation", dataset.get("test", None))
    if test_dataset is not None:
        test_process = RandomProcessImageWithBboxes(target_shape=input_shape, resize_method=resize_method, resize_antialias=resize_antialias, magnitude=-1)
        test_dataset = test_dataset.map(test_process).batch(batch_size, drop_remainder=is_tpu).map(lambda xx, yy: (rescaling(xx), bbox_process(yy)))

    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch


""" Show """


def draw_bboxes(bboxes, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    bboxes = np.array(bboxes).astype("int32")
    for bb in bboxes:
        ax.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])
    plt.show()
    return ax


def show_image_with_bboxes(
    image, bboxes, labels=None, confidences=None, is_bbox_width_first=False, ax=None, label_font_size=8, num_classes=80, indices_2_labels=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    need_plt_show = False
    if ax is None:
        fig, ax = plt.subplots()
        need_plt_show = True
    ax.imshow(image)
    bboxes = np.array(bboxes)
    if is_bbox_width_first:
        bboxes = bboxes[:, [1, 0, 3, 2]]
    for id, bb in enumerate(bboxes):
        # bbox is [top, left, bottom, right]
        bb = [bb[0] * image.shape[0], bb[1] * image.shape[1], bb[2] * image.shape[0], bb[3] * image.shape[1]]
        bb = np.array(bb).astype("int32")
        ax.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])

        if labels is not None:
            label = int(labels[id])
            if indices_2_labels is not None:
                label = indices_2_labels.get(label, indices_2_labels.get(str(label), "None"))
            elif num_classes == 90:
                label = COCO_90_LABEL_DICT[label]
            elif num_classes == 80:
                label = COCO_80_LABEL_DICT[label]

            if confidences is not None:
                label += ": {:.4f}".format(float(confidences[id]))
            color = ax.lines[-1].get_color()
            # ax.text(bb[1], bb[0] - 5, "label: {}, {}".format(label, COCO_80_LABEL_DICT[label]), color=color, fontsize=8)
            ax.text(bb[1], bb[0] - 5, label, color=color, fontsize=label_font_size)
    ax.set_axis_off()
    plt.tight_layout()
    if need_plt_show:
        plt.show()
    return ax


def show_batch_sample(
    dataset, rescale_mode="torch", rows=-1, label_font_size=8, base_size=3, anchors_mode="efficientdet", indices_2_labels=None, **anchor_kwargs
):
    import matplotlib.pyplot as plt
    from keras_cv_attention_models.visualizing import get_plot_cols_rows

    if isinstance(dataset, (list, tuple)):
        images, labels = dataset
    else:
        images, labels = dataset.as_numpy_iterator().next()
    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    images = (images * std + mean) / 255
    if anchors_mode == anchors_func.YOLOR_MODE:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(images.shape[1:-1], labels.shape[1])
        anchors = anchors_func.get_yolor_anchors(images.shape[1:-1], pyramid_levels=pyramid_levels, is_for_training=False)
    elif not anchors_mode == anchors_func.ANCHOR_FREE_MODE:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(images.shape[1:-1], labels.shape[1])
        anchors = anchors_func.get_anchors(images.shape[1:-1], pyramid_levels, **anchor_kwargs)

    cols, rows = get_plot_cols_rows(len(images), rows, ceil_mode=True)
    fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
    axes = axes.flatten()

    for ax, image, label in zip(axes, images, labels):
        if label.shape[-1] == 5:
            pick = label[:, -1] >= 0
            valid_preds = label[pick]
        else:
            pick = label[:, -1] == 1
            valid_preds = label[pick]
            valid_label = tf.cast(tf.argmax(valid_preds[:, 4:-1], axis=-1), valid_preds.dtype)
            valid_preds = tf.concat([valid_preds[:, :4], tf.expand_dims(valid_label, -1)], axis=-1)

        if anchors_mode == anchors_func.YOLOR_MODE:
            valid_anchors = anchors[pick]
            decoded_centers = (valid_preds[:, :2] + 0.5) * valid_anchors[:, 4:] + valid_anchors[:, :2]
            decoded_hw = valid_preds[:, 2:4] * valid_anchors[:, 4:]
            decoded_corner = anchors_func.center_yxhw_to_corners_nd(tf.concat([decoded_centers, decoded_hw], axis=-1))
            valid_preds = tf.concat([decoded_corner, valid_preds[:, -1:]], axis=-1)
        elif not anchors_mode == anchors_func.ANCHOR_FREE_MODE:
            valid_anchors = anchors[pick]
            valid_preds = anchors_func.decode_bboxes(valid_preds, valid_anchors)
        show_image_with_bboxes(image, valid_preds[:, :4], valid_preds[:, -1], ax=ax, label_font_size=label_font_size, indices_2_labels=indices_2_labels)
    fig.tight_layout()
    plt.show()
    return fig
