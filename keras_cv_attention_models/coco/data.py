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

""" Anchors and bboxes """


def get_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 7], aspect_ratios=[1, 2, 0.5], num_scales=3, anchor_scale=4, grid_zero_start=False):
    """
    >>> from keras_cv_attention_models.coco import data
    >>> input_shape = [512, 128]
    >>> anchors = data.get_anchors([512, 128], pyramid_levels=[7])
    >>> data.draw_bboxes(anchors * [512, 128, 512, 128])

    grid_zero_start: grid starts from 0, else from strides // 2. False for efficientdet anchors, True for yolo anchors.
    """
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios = tf.convert_to_tensor(aspect_ratios, dtype="float32")
    if len(aspect_ratios.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = tf.sqrt(aspect_ratios)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = aspect_ratios[:, 0], aspect_ratios[:, 1]
    base_anchors_hh = tf.reshape(tf.expand_dims(scales, 1) * tf.expand_dims(hh_ratios, 0), [-1])
    base_anchors_ww = tf.reshape(tf.expand_dims(scales, 1) * tf.expand_dims(ww_ratios, 0), [-1])
    base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, base_anchors_ww / 2
    base_anchors = tf.stack([base_anchors_hh_half * -1, base_anchors_ww_half * -1, base_anchors_hh_half, base_anchors_ww_half], axis=1)
    # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))

    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feature_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feature_sizes[-1]
        feature_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1))  # ceil mode, like padding="SAME" downsampling

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        top, left = (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        hh_centers = tf.range(top, input_shape[0], stride_hh)
        ww_centers = tf.range(left, input_shape[1], stride_ww)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.reshape(tf.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4])
        anchors = tf.expand_dims(base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0) + tf.cast(grid, base_anchors.dtype)
        anchors = tf.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = tf.concat(all_anchors, axis=0) / [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]
    # if width_first:
    #      all_anchors = tf.gather(all_anchors, [1, 0, 3, 2], axis=-1)
    return all_anchors


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


def assign_anchor_classes_by_iou_with_bboxes(bbox_labels, anchors, ignore_threshold=0.4, overlap_threshold=0.5):
    num_anchors = anchors.shape[0]
    valid_bboxes_pick = tf.where(bbox_labels[:, -1] > 0)[:, 0]
    bbox_labels = tf.gather(bbox_labels, valid_bboxes_pick)
    bboxes, labels = bbox_labels[:, :4], bbox_labels[:, 4]

    anchor_ious = iou_nd(bboxes, anchors)
    anchor_best_iou_ids = tf.argmax(anchor_ious, axis=0)
    # anchor_best_ious = tf.gather_nd(anchor_ious, tf.stack([anchor_best_iou_ids, tf.range(num_anchors, dtype=anchor_best_iou_ids.dtype)], axis=-1))
    anchor_best_ious = tf.reduce_max(anchor_ious, axis=0)  # This faster

    matched_idxes = tf.where(anchor_best_ious > overlap_threshold)[:, 0]
    matched_idxes = tf.unique(tf.concat([matched_idxes, tf.argmax(anchor_ious, axis=-1)], axis=0))[0]  # Ensure at leat one anchor selected for each bbox
    matched_idxes_nd = tf.expand_dims(matched_idxes, -1)
    best_match_indxes = tf.gather(anchor_best_iou_ids, matched_idxes)
    best_match_labels = tf.gather(labels, best_match_indxes)

    # Mark anchors classes, iou < ignore_threshold as 0, ignore_threshold < iou < overlap_threshold as -1
    anchor_classes = tf.where(anchor_best_ious > ignore_threshold, tf.cast(-1, bbox_labels.dtype), tf.cast(0, bbox_labels.dtype))
    # Mark matched anchors classes, iou > overlap_threshold as actual labels
    # anchor_classes = tf.where(anchor_best_ious > overlap_threshold, labels[anchor_best_iou_ids], anchor_classes)
    anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, matched_idxes_nd, tf.cast(best_match_labels, bbox_labels.dtype))

    valid_anchors = tf.gather(anchors, matched_idxes)
    valid_anchors_center, valid_anchors_wh = corners_to_center_xywh_nd(valid_anchors)
    bboxes_center, bboxes_wh = corners_to_center_xywh_nd(bboxes)
    bboxes_centers, bboxes_whs = tf.gather(bboxes_center, best_match_indxes), tf.gather(bboxes_wh, best_match_indxes)

    encoded_anchors_center = (bboxes_centers - valid_anchors_center) / valid_anchors_wh
    encoded_anchors_wh = tf.math.log(bboxes_whs / valid_anchors_wh)
    encoded_anchors = tf.concat([encoded_anchors_center, encoded_anchors_wh], axis=-1)

    dest_boxes = tf.zeros_like(anchors)
    dest_boxes = tf.tensor_scatter_nd_update(dest_boxes, matched_idxes_nd, encoded_anchors)

    rr = tf.concat([dest_boxes, tf.expand_dims(anchor_classes, -1)], axis=-1)
    return rr


def decode_bboxes(preds, anchors):
    bboxes, label = preds[:, :4], preds[:, 4:]
    anchors_wh = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    bboxes_center = bboxes[:, :2] * anchors_wh + anchors_center
    bboxes_wh = tf.math.exp(bboxes[:, 2:]) * anchors_wh

    preds_left_top = bboxes_center - 0.5 * bboxes_wh
    pred_right_bottom = preds_left_top + bboxes_wh
    return tf.concat([preds_left_top, pred_right_bottom, label], axis=-1)


""" Bboxes augment """


def resize_and_crop_bboxes(bboxes, source_shape, target_shape, scale=1.0, offset_y=0, offset_x=0):
    height, width = tf.cast(source_shape[0], bboxes.dtype), tf.cast(source_shape[1], bboxes.dtype)
    # print(height, width, scale)
    if isinstance(scale, (list, tuple)):
        scaled_height, scaled_width = height * scale[0], width * scale[1]
    else:
        scaled_height, scaled_width = height * scale, width * scale
    bboxes *= [scaled_height, scaled_width, scaled_height, scaled_width]

    offset_y, offset_x = tf.cast(offset_y, bboxes.dtype), tf.cast(offset_x, bboxes.dtype)
    bboxes -= [offset_y, offset_x, offset_y, offset_x]
    bboxes /= [target_shape[0], target_shape[1], target_shape[0], target_shape[1]]
    return bboxes


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


def refine_bboxes_labels_batch(bboxes, labels=None, clip_min=0, clip_max=1):
    is_concated = False
    if labels is None:
        bboxes, labels = bboxes[:, :, :4], bboxes[:, :, 4:]
        is_concated = True
    bboxes = tf.clip_by_value(bboxes, clip_min, clip_max)

    mask_cond = tf.not_equal((bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1]), 0)
    # bboxes = tf.where(mask_cond, bboxes, tf.zeros_like(bboxes))
    bboxes = bboxes * tf.expand_dims(tf.cast(mask_cond, bboxes.dtype), -1)
    labels = tf.where(mask_cond, labels, tf.zeros_like(labels))
    return tf.concat([bboxes, labels], axis=-1) if is_concated else (bboxes, labels)


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


def random_crop_and_resize_image(image, target_shape, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), method="bilinear", antialias=False):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    cropped_hh, cropped_ww = random_crop_fraction((height, width), scale, ratio)
    cropped_hh, cropped_ww = tf.clip_by_value(cropped_hh, 1, height - 1), tf.clip_by_value(cropped_ww, 1, width - 1)
    crop_hh = tf.random.uniform((), 0, height - cropped_hh, dtype=cropped_hh.dtype)
    crop_ww = tf.random.uniform((), 0, width - cropped_ww, dtype=cropped_ww.dtype)
    image = image[crop_hh : crop_hh + cropped_hh, crop_ww : crop_ww + cropped_ww]
    image = tf.image.resize(image, target_shape, method=method, antialias=antialias)

    scale_hh = tf.cast(target_shape[0], "float32") / tf.cast(cropped_hh, "float32")
    scale_ww = tf.cast(target_shape[1], "float32") / tf.cast(cropped_ww, "float32")
    crop_hh = tf.cast(tf.cast(crop_hh, "float32") * scale_hh, "int32")
    crop_ww = tf.cast(tf.cast(crop_ww, "float32") * scale_ww, "int32")
    scale = (scale_hh, scale_ww)
    return image, scale, crop_hh, crop_ww


def get_image_aspect_aware_random_scale_crop(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
    """ https://github.com/google/automl/tree/master/efficientdet/dataloader.py#L67 """
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


def aspect_aware_resize_and_crop_image(image, target_shape, scale=-1, crop_y=0, crop_x=0, method="bilinear", antialias=False):
    height, width = tf.cast(tf.shape(image)[0], "float32"), tf.cast(tf.shape(image)[1], "float32")
    if scale == -1:
        scale = tf.minimum(target_shape[0] / height, target_shape[1] / width)
    scaled_hh, scaled_ww = int(height * scale), int(width * scale)
    image = tf.image.resize(image, [scaled_hh, scaled_ww], method=method, antialias=antialias)
    image = image[crop_y : crop_y + target_shape[0], crop_x : crop_x + target_shape[1]]
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_shape[0], target_shape[1])
    return image, scale


def random_flip_left_right_with_bboxes(image, bboxes, probability=0.5):
    # For box, left = 1 - right, right = 1 - left
    return tf.cond(
        tf.random.uniform(()) < probability,
        lambda: (tf.image.flip_left_right(image), tf.gather(bboxes, [0, 3, 2, 1], axis=1) * [1, -1, 1, -1] + [0, 1, 0, 1]),
        lambda: (image, bboxes),
    )


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
        cur_bboxes = resize_and_crop_bboxes(cur_bboxes, (hh, ww), (1, 1), scale, offset_y=crop_hh, offset_x=crop_ww)  # Don't scale to [0, 1]
        sub_hh, sub_ww = tf.cast(sub_hh, cur_bboxes.dtype), tf.cast(sub_ww, cur_bboxes.dtype)
        cur_bboxes, cur_labels = refine_bboxes_labels_batch(cur_bboxes, cur_labels, clip_max=[sub_hh, sub_ww, sub_hh, sub_ww])
        cur_bboxes = cur_bboxes + [top, left, top, left]
        mixed_bboxes.append(cur_bboxes)
        mixed_labels.append(cur_labels)

    top_images = tf.concat([mixed_images[0], mixed_images[1]], axis=2)
    bottom_images = tf.concat([mixed_images[2], mixed_images[3]], axis=2)
    mixed_images = tf.concat([top_images, bottom_images], axis=1)
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
        random_crop_min=1.0,
        resize_method="bilinear",
        resize_antialias=False,
        magnitude=0,
        num_layers=2,
        use_color_increasing=True,
        **randaug_kwargs,
    ):
        self.max_labels_per_image = max_labels_per_image
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.resize_method, self.resize_antialias, self.random_crop_min, self.magnitude = resize_method, resize_antialias, random_crop_min, magnitude
        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            print(">>>> RandAugment: magnitude = %d" % magnitude)
            self.randaug_wo_pos = augment.RandAugment(
                num_layers=num_layers,
                magnitude=magnitude,
                use_cutout=False,
                use_color_increasing=use_color_increasing,
                use_positional_related_ops=False,  # Set False to exlude [shear, rotate, translate]
                **randaug_kwargs,
            )
            # RandAugment positional related ops. Including [shear, rotate, translate], Also returns affine transform matrix
            self.pos_randaug = augment.PositionalRandAugment(num_layers=num_layers, magnitude=magnitude, **randaug_kwargs)

    def __call__(self, datapoint):
        image = datapoint["image"]
        objects = datapoint["objects"]
        bbox, label, is_not_crowd = tf.cast(objects["bbox"], tf.float32), objects["label"], objects["is_crowd"] == False
        bbox, label = tf.boolean_mask(bbox, is_not_crowd), tf.boolean_mask(label, is_not_crowd)
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        if self.magnitude >= 0:
            image, bbox = random_flip_left_right_with_bboxes(image, bbox)
        if self.random_crop_min > 0 and self.random_crop_min < 1:
            # scale, crop_hh, crop_ww = get_image_aspect_aware_random_scale_crop((height, width), self.target_shape)
            # image, scale = aspect_aware_resize_and_crop_image(
            #     image, self.target_shape, scale, crop_hh, crop_ww, method=self.resize_method, antialias=self.resize_antialias
            # )
            # image, scale, crop_hh, crop_ww = random_largest_crop_and_resize_images(image, self.target_shape, self.resize_method, self.resize_antialias)
            image, scale, crop_hh, crop_ww = random_crop_and_resize_image(
                image, self.target_shape, scale=(self.random_crop_min, 1.0), method=self.resize_method, antialias=self.resize_antialias
            )
            bbox = resize_and_crop_bboxes(bbox, (height, width), self.target_shape, scale=scale, offset_y=crop_hh, offset_x=crop_ww)
            bbox, label = refine_bboxes_labels_single(bbox, label)
        else:
            image, scale = aspect_aware_resize_and_crop_image(image, self.target_shape, method=self.resize_method, antialias=self.resize_antialias)
            bbox = resize_and_crop_bboxes(bbox, (height, width), self.target_shape, scale=scale)

        if self.magnitude > 0:
            image.set_shape([*self.target_shape[:2], 3])
            image = self.randaug_wo_pos(image)
            image, affine_matrix = self.pos_randaug(image)
            bbox = bboxes_apply_affine(bbox, affine_matrix, input_shape=image.shape)
            bbox, label = refine_bboxes_labels_single(bbox, label)

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


def to_one_hot_with_class_mark(anchor_bboxes_with_label, num_classes=80):
    dest_boxes, anchor_classes = anchor_bboxes_with_label[:, :4], anchor_bboxes_with_label[:, -1]
    one_hot_labels = tf.one_hot(tf.cast(anchor_classes, "int32") - 1, num_classes)  # [1, 81] -> [0, 80]
    # Mark iou < ignore_threshold as 0, ignore_threshold < iou < overlap_threshold as -1, iou > overlap_threshold as 1
    marks = tf.where(anchor_classes > 0, tf.ones_like(anchor_classes), anchor_classes)
    marks = tf.expand_dims(marks, -1)
    one_hot_labels, marks = tf.cast(one_hot_labels, dest_boxes.dtype), tf.cast(marks, dest_boxes.dtype)
    return tf.concat([dest_boxes, one_hot_labels, marks], axis=-1)


def __bboxes_labels_batch_func__(bboxes, labels, anchors, empty_label, num_classes=80):
    bbox_labels = tf.concat([bboxes, tf.cast(tf.expand_dims(labels, -1), bboxes.dtype)], axis=-1)
    bbox_process = lambda xx: tf.cond(
        tf.math.reduce_any(xx[:, -1] > 0),
        lambda: to_one_hot_with_class_mark(assign_anchor_classes_by_iou_with_bboxes(xx, anchors), num_classes),
        lambda: empty_label,
    )
    return tf.map_fn(bbox_process, bbox_labels)


def init_dataset(
    data_name="coco/2017",  # dataset params
    input_shape=(256, 256),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    max_labels_per_image=100,
    mosaic_mix_prob=0.0,
    anchor_pyramid_levels=[3, 7],
    anchor_aspect_ratios=[1, 2, 0.5],  # [1, 2, 0.5] matches efficientdet anchors format.
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
        max_labels_per_image=max_labels_per_image,
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        magnitude=magnitude,
        num_layers=num_layers,
        **augment_kwargs,
    )
    train_dataset = dataset["train"].shuffle(buffer_size).map(train_process).batch(batch_size)
    # return train_dataset

    if mosaic_mix_prob > 0:
        mosaic_mix = lambda xx, yy: tf.cond(
            tf.random.uniform(()) > mosaic_mix_prob,
            # lambda: (xx, tf.pad(yy[0])),
            lambda: (xx, yy),
            lambda: mosaic_mix_batch(xx, yy[0], yy[1]),
        )
        train_dataset = train_dataset.map(mosaic_mix, num_parallel_calls=AUTOTUNE)
    # return train_dataset

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std
    empty_label = tf.zeros([num_anchors, 4 + num_classes + 1])  # All 0
    bbox_process = lambda bb: __bboxes_labels_batch_func__(bb[0], bb[1], anchors, empty_label, num_classes)

    train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), bbox_process(yy)), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # return train_dataset

    """ Test dataset """
    test_dataset = dataset.get("validation", dataset.get("test", None))
    if test_dataset is not None:
        test_process = RandomProcessImageWithBboxes(target_shape=input_shape, resize_method=resize_method, resize_antialias=resize_antialias, magnitude=-1)
        test_dataset = test_dataset.map(test_process).batch(batch_size).map(lambda xx, yy: (rescaling(xx), bbox_process(yy)))
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
