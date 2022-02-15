import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras_cv_attention_models.imagenet.data import init_mean_std_by_rescale_mode, random_crop_fraction

COCO_LABELS = """person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign,
    parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie,
    suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
    bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
    cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
    toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""
COCO_80_LABEL_DICT = {id: ii.strip() for id, ii in enumerate(COCO_LABELS.split(","))}
INVALID_ID_90 = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
COCO_90_LABEL_DICT = {id: ii for id, ii in zip(set(range(90)) - set(INVALID_ID_90), COCO_80_LABEL_DICT.values())}
COCO_90_LABEL_DICT.update({ii: "Unknown" for ii in INVALID_ID_90})
COCO_80_to_90_LABEL_DICT = {id_80: id_90 for id_80, id_90 in enumerate(set(range(90)) - set(INVALID_ID_90))}


def get_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 7], aspect_ratios=[0.5, 1, 2], num_scales=3, anchor_scale=4):
    """
    >>> from keras_cv_attention_models.coco import data
    >>> input_shape = [512, 128]
    >>> anchors = data.get_anchors([512, 128], pyramid_levels=[7])
    >>> data.draw_bboxes(anchors * [512, 128, 512, 128])
    """
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios = tf.convert_to_tensor(aspect_ratios)
    if len(aspect_ratios.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = tf.sqrt(aspect_ratios)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = aspect_ratios[:, 0], aspect_ratios[:, 1]
    base_anchors_hh = tf.reshape(tf.expand_dims(scales, 0) * tf.expand_dims(hh_ratios, 1), [-1])
    base_anchors_ww = tf.reshape(tf.expand_dims(scales, 0) * tf.expand_dims(ww_ratios, 1), [-1])
    base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, base_anchors_ww / 2
    base_anchors = tf.stack([base_anchors_hh_half * -1, base_anchors_ww_half * -1, base_anchors_hh_half, base_anchors_ww_half], axis=1)
    base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))

    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feat_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feat_sizes[-1]
        feat_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1))

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = feat_sizes[0][0] / feat_sizes[level][0], feat_sizes[0][1] / feat_sizes[level][1]
        hh_centers = tf.range(stride_hh / 2, input_shape[0], stride_hh)
        ww_centers = tf.range(stride_ww / 2, input_shape[1], stride_ww)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.reshape(tf.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4])
        anchors = tf.expand_dims(base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0) + tf.cast(grid, base_anchors.dtype)
        anchors = tf.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    return tf.concat(all_anchors, axis=0) / [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]


def iou_nd(bboxes, anchors):
    anchors_nd, bboxes_nd = tf.expand_dims(anchors, 0), tf.expand_dims(bboxes, 1)
    inter_top_left = tf.maximum(anchors_nd[:, :, :2], bboxes_nd[:, :, :2])
    inter_bottom_right = tf.minimum(anchors_nd[:, :, 2:], bboxes_nd[:, :, 2:])
    inter_wh = tf.maximum(inter_bottom_right - inter_top_left, 0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    union_area = (tf.expand_dims(bboxes_area, 1) + tf.expand_dims(anchors_area, 0)) - inter_area
    return inter_area / union_area


def corners_to_center_xywh_nd(ss):
    """ input: [top, left, bottom, right], output: [center_h, center_w], [height, width] """
    return (ss[:, :2] + ss[:, 2:]) * 0.5, ss[:, 2:] - ss[:, :2]


def assign_anchor_classes_by_iou_with_bboxes(bboxes, anchors, labels, ignore_threshold=0.4, overlap_threshold=0.5):
    num_anchors = anchors.shape[0]
    anchor_ious = iou_nd(bboxes, anchors)
    anchor_best_iou_ids = tf.argmax(anchor_ious, axis=0)
    # anchor_best_ious = tf.gather_nd(anchor_ious, tf.stack([anchor_best_iou_ids, tf.range(num_anchors, dtype=anchor_best_iou_ids.dtype)], axis=-1))
    anchor_best_ious = tf.reduce_max(anchor_ious, axis=0)  # This faster

    matched_idxes = tf.where(anchor_best_ious > overlap_threshold)[:, 0]
    matched_idxes = tf.unique(tf.concat([matched_idxes, tf.argmax(anchor_ious, axis=-1)], axis=0))[0]  # Ensure at leat one anchor selected for each bbox
    matched_idxes_nd = tf.expand_dims(matched_idxes, -1)
    best_match_indxes = tf.gather(anchor_best_iou_ids, matched_idxes)
    best_match_labels = tf.gather(labels, best_match_indxes)

    # Mark anchors classes, iou < ignore_threshold as -1, ignore_threshold < iou < overlap_threshold as -2
    anchor_classes = tf.where(anchor_best_ious > ignore_threshold, tf.cast(-2, best_match_labels.dtype), tf.cast(-1, best_match_labels.dtype))
    # Mark matched anchors classes, iou > overlap_threshold as actual labels
    # anchor_classes = tf.where(anchor_best_ious > overlap_threshold, labels[anchor_best_iou_ids], anchor_classes)
    anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, matched_idxes_nd, best_match_labels)

    valid_anchors = tf.gather(anchors, matched_idxes)
    valid_anchors_center, valid_anchors_wh = corners_to_center_xywh_nd(valid_anchors)
    bboxes_center, bboxes_wh = corners_to_center_xywh_nd(bboxes)
    bboxes_centers, bboxes_whs = tf.gather(bboxes_center, best_match_indxes), tf.gather(bboxes_wh, best_match_indxes)

    encoded_anchors_center = (bboxes_centers - valid_anchors_center) / valid_anchors_wh
    encoded_anchors_wh = tf.math.log(bboxes_whs / valid_anchors_wh)
    encoded_anchors = tf.concat([encoded_anchors_center, encoded_anchors_wh], axis=-1)

    dest_boxes = tf.zeros_like(anchors)
    dest_boxes = tf.tensor_scatter_nd_update(dest_boxes, matched_idxes_nd, encoded_anchors)

    anchor_classes = tf.expand_dims(tf.cast(anchor_classes, dest_boxes.dtype), -1)
    rr = tf.concat([dest_boxes, anchor_classes], axis=-1)
    return rr


def to_one_hot_with_class_mark(anchor_bboxes_with_label, num_classes=80):
    dest_boxes, anchor_classes = anchor_bboxes_with_label[:, :4], anchor_bboxes_with_label[:, -1]
    one_hot_labels = tf.one_hot(tf.cast(anchor_classes, "int32"), num_classes)
    # Mark iou < ignore_threshold as 0, ignore_threshold < iou < overlap_threshold as -1, iou > overlap_threshold as 1
    marks = tf.where(anchor_classes < 0, anchor_classes + 1, tf.ones_like(anchor_classes))
    marks = tf.expand_dims(marks, -1)
    one_hot_labels, marks = tf.cast(one_hot_labels, dest_boxes.dtype), tf.cast(marks, dest_boxes.dtype)
    return tf.concat([dest_boxes, one_hot_labels, marks], axis=-1)


def random_crop_fraction_with_bboxes(image, bboxes, labels, scale=(0.08, 1.0), ratio=(0.75, 1.3333333)):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    crop_hh, crop_ww = random_crop_fraction((height, width), scale, ratio)
    crop_hh, crop_ww = tf.clip_by_value(crop_hh, 1, height - 1), tf.clip_by_value(crop_ww, 1, width - 1)
    crop_top = tf.random.uniform((), 0, height - crop_hh, dtype=crop_hh.dtype)
    crop_left = tf.random.uniform((), 0, width - crop_ww, dtype=crop_ww.dtype)
    crop_bottom, crop_right = crop_top + crop_hh, crop_left + crop_ww
    image = image[crop_top:crop_bottom, crop_left:crop_right]

    height, width = tf.cast(height, bboxes.dtype), tf.cast(width, bboxes.dtype)
    itop, ileft = tf.cast(crop_top, bboxes.dtype) / height, tf.cast(crop_left, bboxes.dtype) / width
    ibottom, iright = tf.cast(crop_bottom, bboxes.dtype) / height, tf.cast(crop_right, bboxes.dtype) / width
    filter_top_left = tf.logical_and(bboxes[:, 0] < ibottom, bboxes[:, 1] < iright)
    filter_bottom_right = tf.logical_and(bboxes[:, 2] > itop, bboxes[:, 3] > ileft)
    border_filter = tf.logical_and(filter_top_left, filter_bottom_right)
    bboxes, labels = tf.boolean_mask(bboxes, border_filter), tf.boolean_mask(labels, border_filter)

    scale_height, scale_width = height / tf.cast(crop_hh, bboxes.dtype), width / tf.cast(crop_ww, bboxes.dtype)
    bboxes = (bboxes - [itop, ileft, itop, ileft]) * [scale_height, scale_width, scale_height, scale_width]
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return image, bboxes, labels


def get_random_image_scale(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
    random_scale_factor = tf.random.uniform([], scale_min, scale_max)
    scaled_y, scaled_x = random_scale_factor * target_shape[0], random_scale_factor * target_shape[1]
    height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
    return tf.minimum(scaled_y / height, scaled_x / width)


def get_image_aspect_aware_random_scale_crop(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
    """ https://github.com/google/automl/tree/master/efficientdet/dataloader.py#L67 """
    random_image_scale = get_random_image_scale(source_shape, target_shape, scale_min, scale_max)

    # Select non-zero random offset (x, y) if scaled image is larger than self._output_size.
    height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
    scaled_height, scaled_width = height * random_image_scale, width * random_image_scale
    offset_y, offset_x = tf.maximum(0.0, scaled_height - target_shape[0]), tf.maximum(0.0, scaled_width - target_shape[1])
    random_offset_y, random_offset_x = offset_y * tf.random.uniform([], 0, 1), offset_x * tf.random.uniform([], 0, 1)
    random_offset_y, random_offset_x = tf.cast(random_offset_y, tf.int32), tf.cast(random_offset_x, tf.int32)
    return random_image_scale, random_offset_y, random_offset_x


def aspect_aware_resize_and_crop_image(image, target_shape, scale=-1, crop_y=0, crop_x=0, method="bilinear", antialias=False):
    height, width = tf.cast(tf.shape(image)[0], "float32"), tf.cast(tf.shape(image)[1], "float32")
    if scale == -1:
        scale = tf.minimum(target_shape[0] / height, target_shape[1] / width)
    scaled_hh, scaled_ww = int(height * scale), int(width * scale)
    image = tf.image.resize(image, [scaled_hh, scaled_ww], method=method, antialias=antialias)
    image = image[crop_y : crop_y + target_shape[0], crop_x : crop_x + target_shape[1]]
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_shape[0], target_shape[1])
    return image, scale


def resize_and_crop_bboxes(bboxes, labels, source_shape, target_shape, scale, offset_y, offset_x):
    height, width = tf.cast(source_shape[0], bboxes.dtype), tf.cast(source_shape[1], bboxes.dtype)
    scaled_height, scaled_width = height * scale, width * scale
    bboxes *= [scaled_height, scaled_width, scaled_height, scaled_width]

    offset_y, offset_x = tf.cast(offset_y, bboxes.dtype), tf.cast(offset_x, bboxes.dtype)
    bboxes -= [offset_y, offset_x, offset_y, offset_x]
    bboxes /= [target_shape[0], target_shape[1], target_shape[0], target_shape[1]]
    bboxes = tf.clip_by_value(bboxes, 0, 1)

    picking_indices = tf.where(tf.not_equal((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]), 0))
    return tf.gather_nd(bboxes, picking_indices), tf.gather_nd(labels, picking_indices)


def random_flip_left_right_with_bboxes(image, bboxes, probability=0.5):
    # For box, left = 1 - right, right = 1 - left
    return tf.cond(
        tf.random.uniform(()) < probability,
        lambda: (tf.image.flip_left_right(image), tf.gather(bboxes, [0, 3, 2, 1], axis=1) * [1, -1, 1, -1] + [0, 1, 0, 1]),
        lambda: (image, bboxes),
    )


class RandomProcessImageWithBboxes:
    def __init__(
        self,
        target_shape=(300, 300),
        random_crop_min=1.0,
        resize_method="bilinear",
        resize_antialias=False,
        magnitude=0,
        num_layers=2,
        use_color_increasing=True,
        **randaug_kwargs,
    ):
        self.magnitude = magnitude
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.resize_method, self.resize_antialias, self.random_crop_min = resize_method, resize_antialias, random_crop_min
        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            print(">>>> RandAugment: magnitude = %d" % magnitude)
            self.randaug = augment.RandAugment(
                num_layers=num_layers,
                magnitude=magnitude,
                use_cutout=False,
                use_color_increasing=use_color_increasing,
                use_positional_related_ops=False,  # Set False to exlude [shear, rotate, translate]
                **randaug_kwargs,
            )

    def __call_1__(self, datapoint):
        image = datapoint["image"]
        objects = datapoint["objects"]
        bbox, label, is_not_crowd = tf.cast(objects["bbox"], tf.float32), objects["label"], objects["is_crowd"] == False
        bbox, label = tf.boolean_mask(bbox, is_not_crowd), tf.boolean_mask(label, is_not_crowd)
        if self.random_crop_min > 0 and self.random_crop_min < 1:
            image, bbox, label = random_crop_fraction_with_bboxes(image, bbox, label, scale=(self.random_crop_min, 1.0))
        if self.magnitude >= 0:
            image, bbox, label = random_flip_left_right_with_bboxes(image, bbox, label)
        if self.magnitude > 0:
            image = self.randaug(image)

        input_image = tf.image.resize(image, self.target_shape, method=self.resize_method, antialias=self.resize_antialias)
        input_image = tf.cast(input_image, tf.float32)
        input_image.set_shape([*self.target_shape[:2], 3])

        return input_image, (bbox, label)

    def __call__(self, datapoint):
        image = datapoint["image"]
        objects = datapoint["objects"]
        bbox, label, is_not_crowd = tf.cast(objects["bbox"], tf.float32), objects["label"], objects["is_crowd"] == False
        bbox, label = tf.boolean_mask(bbox, is_not_crowd), tf.boolean_mask(label, is_not_crowd)

        if self.magnitude >= 0:
            processed_image, bbox = random_flip_left_right_with_bboxes(processed_image, bbox)
        if self.random_crop_min > 0 and self.random_crop_min < 1:
            scale, offset_y, offset_x = get_image_aspect_aware_random_scale_crop(tf.shape(image), self.target_shape)
        else:
            scale, offset_y, offset_x = -1, 0, 0  # Evaluation
        processed_image, scale = aspect_aware_resize_and_crop_image(
            image, self.target_shape, scale, offset_y, offset_x, method=self.resize_method, antialias=self.resize_antialias
        )
        bbox, label = resize_and_crop_bboxes(bbox, label, tf.shape(image), self.target_shape, scale, offset_y, offset_x)

        if self.magnitude > 0:
            processed_image = self.randaug(processed_image)

        processed_image = tf.cast(processed_image, tf.float32)
        processed_image.set_shape([*self.target_shape[:2], 3])

        return processed_image, (bbox, label)


def init_dataset(
    data_name="coco/2017",  # dataset params
    input_shape=(256, 256),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    anchor_pyramid_levels=[3, 7],
    anchor_aspect_ratios=[0.5, 1, 2],
    anchor_num_scales=3,
    anchor_scale=4,
    rescale_mode="torch",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    random_crop_min=1.0,
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    resize_antialias=False,
    magnitude=0,
    num_layers=2,
    **augment_kwargs,  # Too many...
):
    try_gcs = True if len(tf.config.list_logical_devices("TPU")) > 0 else False
    dataset, info = tfds.load(data_name, with_info=True, try_gcs=try_gcs)
    num_classes = info.features["objects"]["label"].num_classes
    total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    anchors = get_anchors(input_shape[:2], anchor_pyramid_levels, anchor_aspect_ratios, anchor_num_scales, anchor_scale)
    num_anchors = anchors.shape[0]

    train_process = RandomProcessImageWithBboxes(
        target_shape=input_shape,
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        magnitude=magnitude,
        num_layers=num_layers,
        **augment_kwargs,
    )
    empty_label = tf.zeros([num_anchors, 4 + num_classes + 1])  # All 0
    bbox_process = lambda bbox, label: tf.cond(
        tf.shape(bbox)[0] == 0,
        lambda: empty_label,
        lambda: to_one_hot_with_class_mark(assign_anchor_classes_by_iou_with_bboxes(bbox, anchors, label), num_classes),
    )
    train_dataset = dataset["train"].shuffle(buffer_size).map(train_process).map(lambda xx, yy: (xx, bbox_process(yy[0], yy[1])))
    train_dataset = train_dataset.batch(batch_size)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std
    train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), yy), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    """ Test dataset """
    test_dataset = dataset.get("validation", dataset.get("test", None))
    if test_dataset is not None:
        test_process = RandomProcessImageWithBboxes(target_shape=input_shape, resize_method=resize_method, resize_antialias=resize_antialias, magnitude=-1)
        test_dataset = test_dataset.map(test_process).map(lambda xx, yy: (xx, bbox_process(yy[0], yy[1])))
        test_dataset = test_dataset.batch(batch_size).map(lambda xx, yy: (rescaling(xx), yy))
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch


def decode_bboxes(preds, anchors):
    bboxes, label = preds[:, :4], preds[:, 4:]
    anchors_wh = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    bboxes_center = bboxes[:, :2] * anchors_wh + anchors_center
    bboxes_wh = tf.math.exp(bboxes[:, 2:]) * anchors_wh

    preds_left_top = bboxes_center - 0.5 * bboxes_wh
    pred_right_bottom = preds_left_top + bboxes_wh
    return tf.concat([preds_left_top, pred_right_bottom, label], axis=-1)


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


def show_image_with_bboxes(image, bboxes, labels=None, confidences=None, ax=None, label_font_size=8, num_classes=80):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    bboxes = np.array(bboxes)
    for id, bb in enumerate(bboxes):
        # bbox is [top, left, bottom, right]
        bb = [bb[0] * image.shape[0], bb[1] * image.shape[1], bb[2] * image.shape[0], bb[3] * image.shape[1]]
        bb = np.array(bb).astype("int32")
        ax.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])

        if labels is not None:
            label = int(labels[id])
            label = COCO_90_LABEL_DICT[label] if num_classes == 90 else COCO_80_LABEL_DICT[label]
            if confidences is not None:
                label += ": {:.4f}".format(float(confidences[id]))
            color = ax.lines[-1].get_color()
            # ax.text(bb[1], bb[0] - 5, "label: {}, {}".format(label, COCO_80_LABEL_DICT[label]), color=color, fontsize=8)
            ax.text(bb[1], bb[0] - 5, label, color=color, fontsize=label_font_size)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return ax


def show_batch_sample(dataset, rescale_mode="torch", rows=-1, label_font_size=8, base_size=3, **anchor_kwargs):
    import matplotlib.pyplot as plt
    from keras_cv_attention_models.visualizing import get_plot_cols_rows

    if isinstance(dataset, (list, tuple)):
        images, labels = dataset
    else:
        images, labels = dataset.as_numpy_iterator().next()
    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    images = (images * std + mean) / 255
    anchors = get_anchors(images.shape[1:-1], **anchor_kwargs)

    cols, rows = get_plot_cols_rows(len(images), rows, ceil_mode=True)
    fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
    axes = axes.flatten()

    for ax, image, label in zip(axes, images, labels):
        if label.shape[-1] == 5:
            pick = label[:, -1] >= 0
            valid_preds, valid_anchors = label[pick], anchors[pick]
        else:
            pick = label[:, -1] == 1
            valid_preds, valid_anchors = label[pick], anchors[pick]
            valid_label = tf.cast(tf.argmax(valid_preds[:, 4:-1], axis=-1), valid_preds.dtype)
            valid_preds = tf.concat([valid_preds[:, :4], tf.expand_dims(valid_label, -1)], axis=-1)

        valid_label = decode_bboxes(valid_preds, valid_anchors)
        show_image_with_bboxes(image, valid_label[:, :4], valid_label[:, -1], ax=ax, label_font_size=label_font_size)
    fig.tight_layout()
    plt.show()
    return fig
