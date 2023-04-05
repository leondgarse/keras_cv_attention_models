import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional

if backend.is_tensorflow_backend:
    import tensorflow as tf  # Needed for all assigning achors functions


EFFICIENTDET_MODE = "efficientdet"
ANCHOR_FREE_MODE = "anchor_free"
YOLOR_MODE = "yolor"
YOLOV8_MODE = "yolov8"
NUM_ANCHORS = {ANCHOR_FREE_MODE: 1, YOLOV8_MODE: 1, YOLOR_MODE: 3, EFFICIENTDET_MODE: 9}

""" Init anchors """


def get_anchors_mode_parameters(anchors_mode, use_object_scores="auto", num_anchors="auto", anchor_scale="auto"):
    if anchors_mode == ANCHOR_FREE_MODE:
        use_object_scores = True if use_object_scores == "auto" else use_object_scores
        num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
    elif anchors_mode == YOLOR_MODE:
        use_object_scores = True if use_object_scores == "auto" else use_object_scores
        num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
    elif anchors_mode == YOLOV8_MODE:
        use_object_scores = False if use_object_scores == "auto" else use_object_scores
        num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
    else:
        use_object_scores = False if use_object_scores == "auto" else use_object_scores
        num_anchors = NUM_ANCHORS.get(anchors_mode, NUM_ANCHORS[EFFICIENTDET_MODE]) if num_anchors == "auto" else num_anchors
        anchor_scale = 4 if anchor_scale == "auto" else anchor_scale
    return use_object_scores, num_anchors, anchor_scale


def get_feature_sizes(input_shape, pyramid_levels=[3, 7]):
    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feature_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feature_sizes[-1]
        feature_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1))  # ceil mode, like padding="SAME" downsampling
    return feature_sizes


def get_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 7], aspect_ratios=[1, 2, 0.5], num_scales=3, anchor_scale=4, grid_zero_start=False):
    """
    >>> from keras_cv_attention_models.coco import anchors_func
    >>> input_shape = [512, 128]
    >>> anchors = anchors_func.get_anchors([512, 128], pyramid_levels=[7])
    >>> anchors.draw_bboxes(anchors * [512, 128, 512, 128])

    grid_zero_start: grid starts from 0, else from strides // 2. False for efficientdet anchors, True for yolo anchors.
    """
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios_tensor = np.array(aspect_ratios, dtype="float32")
    if len(aspect_ratios_tensor.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = np.sqrt(aspect_ratios_tensor)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = aspect_ratios_tensor[:, 0], aspect_ratios_tensor[:, 1]
    base_anchors_hh = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(hh_ratios, 0), [-1])
    base_anchors_ww = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(ww_ratios, 0), [-1])
    base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, base_anchors_ww / 2
    base_anchors = np.stack([base_anchors_hh_half * -1, base_anchors_ww_half * -1, base_anchors_hh_half, base_anchors_ww_half], axis=1)
    # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        top, left = (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        hh_centers = np.arange(top, input_shape[0], stride_hh)
        ww_centers = np.arange(left, input_shape[1], stride_ww)
        ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
        grid = np.reshape(np.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4])
        anchors = np.expand_dims(base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0) + grid.astype(base_anchors.dtype)
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = np.concatenate(all_anchors, axis=0) / [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]
    # if width_first:
    #      all_anchors = tf.gather(all_anchors, [1, 0, 3, 2], axis=-1)

    return functional.convert_to_tensor(all_anchors.astype("float32"))


def get_anchor_free_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 5], grid_zero_start=True):
    return get_anchors(input_shape, pyramid_levels, aspect_ratios=[1], num_scales=1, anchor_scale=1, grid_zero_start=grid_zero_start)


def get_yolor_anchors(input_shape=(512, 512), pyramid_levels=[3, 5], offset=0.5, is_for_training=False):
    assert max(pyramid_levels) - min(pyramid_levels) < 5
    # Original yolor using width first, height first here
    if max(pyramid_levels) - min(pyramid_levels) < 3:  # [3, 5], YOLOR_CSP / YOLOR_CSPX
        anchor_ratios = np.array([[[16.0, 12], [36, 19], [28, 40]], [[75, 36], [55, 76], [146, 72]], [[110, 142], [243, 192], [401, 459]]])
        # anchor_ratios = tf.convert_to_tensor([[[13.0, 10], [30, 16], [23, 33]], [[61, 30], [45, 62], [119, 59]], [[90, 116], [198, 156], [326, 373]]])
    elif max(pyramid_levels) - min(pyramid_levels) < 4:  # [3, 6], YOLOR_*6
        anchor_ratios = np.array(
            [[[27.0, 19], [40, 44], [94, 38]], [[68, 96], [152, 86], [137, 180]], [[301, 140], [264, 303], [542, 238]], [[615, 436], [380, 739], [792, 925]]]
        )
    else:  # [3, 7] from YOLOV4_P7, using first 3 for each level
        anchor_ratios = np.array(
            [
                [[17.0, 13], [25, 22], [66, 27]],
                [[88, 57], [69, 112], [177, 69]],
                [[138, 136], [114, 287], [275, 134]],
                [[248, 268], [504, 232], [416, 445]],
                [[393, 812], [808, 477], [908, 1070]],
            ]
        )

    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)
    # print(f"{pyramid_levels = }, {feature_sizes = }, {anchor_ratios = }")
    if is_for_training:
        # YOLOLayer https://github.com/WongKinYiu/yolor/blob/main/models/models.py#L351
        anchor_ratios = anchor_ratios[: len(pyramid_levels)] / [[[2**ii]] for ii in pyramid_levels]
        feature_sizes = np.array(feature_sizes[min(pyramid_levels) : max(pyramid_levels) + 1], "float32")
        return anchor_ratios, feature_sizes

    all_anchors = []
    for level, anchor_ratio in zip(pyramid_levels, anchor_ratios):
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        # hh_grid, ww_grid = tf.meshgrid(tf.range(feature_sizes[level][0]), tf.range(feature_sizes[level][1]))
        ww_grid, hh_grid = np.meshgrid(np.arange(feature_sizes[level][1]), np.arange(feature_sizes[level][0]))
        grid = np.stack([hh_grid, ww_grid], 2).astype("float32") - offset
        grid = np.reshape(grid, [-1, 1, 2])  # [1, level_feature_sizes, 2]
        cur_base_anchors = anchor_ratio[np.newaxis, :, :]  # [num_anchors, 1, 2]

        grid_nd = np.repeat(grid, cur_base_anchors.shape[1], axis=1) * [stride_hh, stride_ww]
        cur_base_anchors_nd = np.repeat(cur_base_anchors, grid.shape[0], axis=0)
        stride_nd = np.zeros_like(grid_nd) + [stride_hh, stride_ww]
        # yield grid_nd, cur_base_anchors_nd, stride_nd
        anchors = np.concatenate([grid_nd, cur_base_anchors_nd, stride_nd], axis=-1)
        all_anchors.append(np.reshape(anchors, [-1, 6]))
    all_anchors = np.concatenate(all_anchors, axis=0) / ([input_shape[0], input_shape[1]] * 3)
    return functional.convert_to_tensor(all_anchors.astype("float32"))  # [center_h, center_w, anchor_h, anchor_w, stride_h, stride_w]


def get_anchors_mode_by_anchors(input_shape, total_anchors, num_anchors="auto", pyramid_levels_min=3, num_anchors_at_each_level_cumsum=None):
    if num_anchors_at_each_level_cumsum is None:
        feature_sizes = get_feature_sizes(input_shape, [pyramid_levels_min, pyramid_levels_min + 10])[pyramid_levels_min:]
        feature_sizes = np.array(feature_sizes, dtype="int32")
        num_anchors_at_each_level_cumsum = np.cumsum(np.prod(feature_sizes, axis=-1))

    if num_anchors == "auto":
        # Pick from [1, 3, 9], 1 for anchor_free, 3 for yolor, 9 for efficientdet
        picks = np.array([1, 3, 9], dtype="int32")
        max_anchors = num_anchors_at_each_level_cumsum[-1] * picks
        num_anchors = np.ceil(total_anchors / max_anchors[0]) if total_anchors > max_anchors[-1] else picks[np.argmax(total_anchors < max_anchors)]
        num_anchors = int(num_anchors)
    dd = {1: ANCHOR_FREE_MODE, 3: YOLOR_MODE, 9: EFFICIENTDET_MODE}
    return dd.get(num_anchors, EFFICIENTDET_MODE), num_anchors


def get_pyramid_levels_by_anchors(input_shape, total_anchors, num_anchors="auto", pyramid_levels_min=3):
    feature_sizes = get_feature_sizes(input_shape, [pyramid_levels_min, pyramid_levels_min + 10])[pyramid_levels_min:]
    feature_sizes = np.array(feature_sizes, dtype="int32")
    num_anchors_at_each_level_cumsum = np.cumsum(np.prod(feature_sizes, axis=-1))
    if num_anchors == "auto":
        _, num_anchors = get_anchors_mode_by_anchors(input_shape, total_anchors, num_anchors, pyramid_levels_min, num_anchors_at_each_level_cumsum)

    total_anchors = total_anchors // num_anchors
    pyramid_levels_max = pyramid_levels_min + np.argmax(num_anchors_at_each_level_cumsum > total_anchors) - 1
    return [pyramid_levels_min, int(pyramid_levels_max)]


""" Bbox functions """


def iou_nd(bboxes, anchors):
    # bboxes: [[top, left, bottom, right]], anchors: [[top, left, bottom, right]]
    anchors_nd, bboxes_nd = functional.expand_dims(anchors, 0), functional.expand_dims(bboxes, 1)
    inter_top_left = functional.maximum(anchors_nd[:, :, :2], bboxes_nd[:, :, :2])
    inter_bottom_right = functional.minimum(anchors_nd[:, :, 2:], bboxes_nd[:, :, 2:])
    inter_hw = functional.maximum(inter_bottom_right - inter_top_left, 0)
    inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]

    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    union_area = (functional.expand_dims(bboxes_area, 1) + functional.expand_dims(anchors_area, 0)) - inter_area
    return inter_area / union_area


def corners_to_center_yxhw_nd(ss):
    """input: [top, left, bottom, right], output: [center_h, center_w], [height, width]"""
    return (ss[:, :2] + ss[:, 2:]) * 0.5, ss[:, 2:] - ss[:, :2]


def center_yxhw_to_corners_nd(ss):
    """input: [center_h, center_w, height, width], output: [top, left, bottom, right]"""
    top_left = ss[:, :2] - ss[:, 2:] * 0.5
    bottom_right = top_left + ss[:, 2:]
    return functional.concat([top_left, bottom_right], axis=-1)


def decode_bboxes(preds, anchors, return_centers=False):
    preds_center, preds_hw, preds_others = functional.split(preds, [2, 2, -1], axis=-1)
    if anchors.shape[-1] == 6:  # Currently, it's yolor anchors
        # anchors: [grid_y, grid_x, base_anchor_y, base_anchor_x, stride_y, stride_x]
        bboxes_center = preds_center * 2 * anchors[:, 4:] + anchors[:, :2]
        bboxes_hw = (preds_hw * 2) ** 2 * anchors[:, 2:4]
    else:
        anchors_hw = anchors[:, 2:] - anchors[:, :2]
        anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

        bboxes_center = preds_center * anchors_hw + anchors_center
        bboxes_hw = functional.exp(preds_hw) * anchors_hw

    if return_centers:
        return functional.concat([bboxes_center, bboxes_hw, preds_others], axis=-1)
    else:
        preds_top_left = bboxes_center - 0.5 * bboxes_hw
        pred_bottom_right = preds_top_left + bboxes_hw
        return functional.concat([preds_top_left, pred_bottom_right, preds_others], axis=-1)


def yolov8_decode_bboxes(preds, anchors, regression_max=16, return_centers=False):
    preds_bbox, preds_others = functional.split(preds, [4 * regression_max, -1], axis=-1)
    preds_bbox = functional.reshape(preds_bbox, [*preds_bbox.shape[:-1], 4, regression_max])
    preds_bbox = functional.softmax(preds_bbox, axis=-1) * functional.range(regression_max, dtype='float32')
    preds_bbox = functional.reduce_sum(preds_bbox, axis=-1)
    preds_top_left, preds_bottom_right = functional.split(preds_bbox, [2, 2], axis=-1)

    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    bboxes_center = (preds_bottom_right - preds_top_left) / 2 * anchors_hw + anchors_center
    bboxes_hw = (preds_bottom_right + preds_top_left) * anchors_hw
    if return_centers:
        return functional.concat([bboxes_center, bboxes_hw, preds_others], axis=-1)
    else:
        preds_top_left = bboxes_center - 0.5 * bboxes_hw
        pred_bottom_right = preds_top_left + bboxes_hw
        return functional.concat([preds_top_left, pred_bottom_right, preds_others], axis=-1)


""" Assign achors """


def assign_anchor_classes_by_iou_with_bboxes(bbox_labels, anchors, ignore_threshold=0.4, overlap_threshold=0.5):
    import tensorflow as tf

    num_anchors = anchors.shape[0]
    bbox_labels = tf.gather_nd(bbox_labels, tf.where(bbox_labels[:, -1] > 0))
    bboxes, labels = bbox_labels[:, :4], bbox_labels[:, 4]

    anchor_ious = iou_nd(bboxes, anchors)  # [num_bboxes, num_anchors]
    anchor_best_iou_ids = tf.argmax(anchor_ious, axis=0)  # [num_anchors]
    # anchor_best_ious = tf.gather_nd(anchor_ious, tf.stack([anchor_best_iou_ids, tf.range(num_anchors, dtype=anchor_best_iou_ids.dtype)], axis=-1))
    anchor_best_ious = tf.reduce_max(anchor_ious, axis=0)  # This faster, [num_anchors]

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
    valid_anchors_center, valid_anchors_hw = corners_to_center_yxhw_nd(valid_anchors)
    bboxes_center, bboxes_hw = corners_to_center_yxhw_nd(bboxes)
    bboxes_centers, bboxes_hws = tf.gather(bboxes_center, best_match_indxes), tf.gather(bboxes_hw, best_match_indxes)

    encoded_anchors_center = (bboxes_centers - valid_anchors_center) / valid_anchors_hw
    encoded_anchors_hw = tf.math.log(bboxes_hws / valid_anchors_hw)
    encoded_anchors = tf.concat([encoded_anchors_center, encoded_anchors_hw], axis=-1)

    dest_boxes = tf.zeros_like(anchors)
    dest_boxes = tf.tensor_scatter_nd_update(dest_boxes, matched_idxes_nd, encoded_anchors)

    rr = tf.concat([dest_boxes, tf.expand_dims(anchor_classes, -1)], axis=-1)
    return rr


def yolor_assign_anchors(bbox_labels, anchor_ratios, feature_sizes, anchor_aspect_thresh=4.0, overlap_offset=0.5):
    """
    # Actual assigning test:
    >>> from keras_cv_attention_models import yolor, test_images
    >>> from keras_cv_attention_models.coco import anchors_func, data
    >>> mm = yolor.YOLOR_CSP()
    >>> img = test_images.dog_cat()
    >>> pred = mm(mm.preprocess_input(img))

    >>> bbs, lls, ccs = mm.decode_predictions(pred)[0]
    >>> bbox_labels = tf.concat([bbs, tf.expand_dims(tf.cast(lls, "float32"), -1)], axis=-1)
    >>> anchor_ratios, feature_sizes = anchors_func.get_yolor_anchors(mm.input_shape[1:-1], pyramid_levels=[3, 5], is_for_training=True)
    >>> assigned = anchors_func.yolor_assign_anchors(bbox_labels, anchor_ratios, feature_sizes)

    >>> # Decode and show matched
    >>> anchors = anchors_func.get_yolor_anchors(mm.input_shape[1:-1], pyramid_levels=[3, 5], is_for_training=False)
    >>> assigned, anchors = assigned[assigned[:, -1] > 0], anchors[assigned[:, -1] > 0]
    >>> decoded_centers = (assigned[:, :2] + 0.5) * anchors[:, 4:] + anchors[:, :2]
    >>> decoded_hw = assigned[:, 2:4] * anchors[:, 4:]  # assigned[:, 2:4] is multiplied with feature_size ==> assigned[:, 2:4] * strides / input_shape
    >>> decoded_corner = anchors_func.center_yxhw_to_corners_nd(tf.concat([decoded_centers, decoded_hw], axis=-1))
    >>> data.show_image_with_bboxes(test_images.dog_cat(), decoded_corner, assigned[:, -1:])
    """
    import tensorflow as tf

    bbox_labels = tf.gather_nd(bbox_labels, tf.where(bbox_labels[:, -1] > 0))
    bboxes, labels = bbox_labels[:, :4], bbox_labels[:, 4:]
    num_anchors, num_bboxes_true, num_output_channels = anchor_ratios.shape[1], tf.shape(bboxes)[0], bbox_labels.shape[-1]

    rrs = []
    # for anchor_ratio, feature_size in zip(anchor_ratios, feature_sizes):
    for id in range(feature_sizes.shape[0]):
        # build_targets https://github.dev/WongKinYiu/yolor/blob/main/utils/loss.py#L127
        anchor_ratio, feature_size = anchor_ratios[id], feature_sizes[id]
        # pick by aspect ratio
        bboxes_centers, bboxes_hws = corners_to_center_yxhw_nd(bboxes)
        bboxes_centers, bboxes_hws = bboxes_centers * feature_size, bboxes_hws * feature_size
        aspect_ratio = tf.expand_dims(bboxes_hws, 0) / tf.expand_dims(anchor_ratio, 1)
        aspect_pick = tf.reduce_max(tf.maximum(aspect_ratio, 1 / aspect_ratio), axis=-1) < anchor_aspect_thresh
        anchors_pick = tf.repeat(tf.expand_dims(tf.range(num_anchors), -1), num_bboxes_true, axis=-1)[aspect_pick]
        aspect_picked_bboxes_labels = tf.concat([bboxes_centers, bboxes_hws, labels], axis=-1)
        aspect_picked_bboxes_labels = tf.repeat(tf.expand_dims(aspect_picked_bboxes_labels, 0), num_anchors, axis=0)[aspect_pick]

        # pick by centers
        centers = aspect_picked_bboxes_labels[:, :2]
        top, left = tf.unstack(tf.logical_and(centers % 1 < overlap_offset, centers > 1), axis=-1)
        bottom, right = tf.unstack(tf.logical_and(centers % 1 > (1 - overlap_offset), centers < (feature_size - 1)), axis=-1)
        anchors_pick_all = tf.concat([anchors_pick, anchors_pick[top], anchors_pick[left], anchors_pick[bottom], anchors_pick[right]], axis=0)
        matched_top, matched_left = aspect_picked_bboxes_labels[top], aspect_picked_bboxes_labels[left]
        matched_bottom, matched_right = aspect_picked_bboxes_labels[bottom], aspect_picked_bboxes_labels[right]
        matched_bboxes_all = tf.concat([aspect_picked_bboxes_labels, matched_top, matched_left, matched_bottom, matched_right], axis=0)

        # Use matched bboxes as indexes
        matched_bboxes_idx = tf.cast(aspect_picked_bboxes_labels[:, :2], "int32")
        matched_top_idx = tf.cast(matched_top[:, :2] - [overlap_offset, 0], "int32")
        matched_left_idx = tf.cast(matched_left[:, :2] - [0, overlap_offset], "int32")
        matched_bottom_idx = tf.cast(matched_bottom[:, :2] + [overlap_offset, 0], "int32")
        matched_right_idx = tf.cast(matched_right[:, :2] + [0, overlap_offset], "int32")
        matched_bboxes_idx_all = tf.concat([matched_bboxes_idx, matched_top_idx, matched_left_idx, matched_bottom_idx, matched_right_idx], axis=0)

        # Results
        centers_true = matched_bboxes_all[:, :2] - tf.cast(matched_bboxes_idx_all, matched_bboxes_all.dtype)
        bbox_labels_true = tf.concat([centers_true, matched_bboxes_all[:, 2:]], axis=-1)
        rr = tf.zeros([feature_size[0], feature_size[1], num_anchors, num_output_channels])
        matched_bboxes_idx_all = tf.clip_by_value(matched_bboxes_idx_all, 0, tf.cast(feature_size, matched_bboxes_idx_all.dtype) - 1)
        rr = tf.tensor_scatter_nd_update(rr, tf.concat([matched_bboxes_idx_all, tf.expand_dims(anchors_pick_all, 1)], axis=-1), bbox_labels_true)
        rrs.append(tf.reshape(rr, [-1, num_output_channels]))
    return tf.concat(rrs, axis=0)


class AnchorFreeAssignMatching:
    """
    This has to be after getting model output, as picking matched anchors needs the iou value between prediction and true bboxes.

    # Basic test:
    >>> from keras_cv_attention_models.coco import anchors_func
    >>> aa = anchors_func.AnchorFreeAssignMatching([640, 640])

    >>> # Fake data
    >>> num_bboxes, num_classes, num_anchors = 32, 10, 8400
    >>> bboxes_true = tf.random.uniform([num_bboxes, 4], 0, 0.5)
    >>> bboxes_true = tf.concat([bboxes_true[:, :2], bboxes_true[:, 2:] + bboxes_true[:, :2]], axis=-1) # bottom, right > top, left
    >>> labels_true = tf.one_hot(tf.random.uniform([num_bboxes], 0, num_classes, dtype=tf.int32), num_classes)
    >>> valid_bboxes_pick = tf.cast(tf.random.uniform([num_bboxes, 1]) > 0.5, tf.float32)
    >>> bbox_labels_true = tf.concat([bboxes_true, labels_true, valid_bboxes_pick], axis=-1)
    >>> bbox_labels_pred = tf.random.uniform([num_anchors, 4 + num_classes + 1])

    >>> # Run test
    >>> bbox_labels_true_assined = aa(bbox_labels_true, bbox_labels_pred)
    >>> bboxes_true, bboxes_true_encoded, labels_true, object_true_idx_nd = tf.split(bbox_labels_true_assined, [4, 4, -1, 1], axis=-1)
    >>> object_true_idx_nd = tf.cast(object_true_idx_nd, tf.int32)
    >>> object_true_idx = object_true_idx_nd[:, 0]
    >>> object_true = tf.tensor_scatter_nd_update(tf.zeros_like(bbox_labels_pred[:, -1]), object_true_idx_nd, tf.ones_like(bboxes_true[:, -1]))
    >>> print(bboxes_true.shape, bboxes_true_encoded.shape, labels_true.shape, tf.reduce_sum(tf.cast(object_true, tf.float32)).numpy())
    >>> # (42, 4), (42, 4), (42, 10), 42.0
    >>> print(f"{object_true.shape = }, {bbox_labels_pred[object_true > 0].shape = }")
    >>> # object_true.shape = TensorShape([8400]), bbox_labels_pred[object_true > 0].shape = TensorShape([42, 15])

    # Actual assigning test:
    >>> from keras_cv_attention_models import yolox, test_images
    >>> from keras_cv_attention_models.coco import anchors_func, data
    >>> mm = yolox.YOLOXS()
    >>> img = test_images.dog_cat()
    >>> pred = mm(mm.preprocess_input(img))

    >>> aa = anchors_func.AnchorFreeAssignMatching([640, 640])
    >>> bbs, lls, ccs = mm.decode_predictions(pred)[0]
    >>> bbox_labels_true = tf.concat([bbs, tf.one_hot(lls, 80), tf.ones([bbs.shape[0], 1])], axis=-1)
    >>> bbox_labels_true_assined = aa(bbox_labels_true, pred[0])
    >>> bboxes_true, bboxes_true_encoded, labels_true, object_true_idx_nd = tf.split(bbox_labels_true_assined, [4, 4, -1, 1], axis=-1)
    >>> object_true_idx_nd = tf.cast(object_true_idx_nd, tf.int32)
    >>> object_true_idx = object_true_idx_nd[:, 0]
    >>> object_true = tf.tensor_scatter_nd_update(tf.zeros_like(pred[0, :, -1]), object_true_idx_nd, tf.ones_like(bboxes_true[:, -1]))
    >>> # Decode predictions
    >>> bbox_labels_pred_valid, anchors = pred[0][object_true > 0], aa.anchors[object_true > 0]
    >>> bboxes_preds, labels_preds, object_preds = bbox_labels_pred_valid[:, :4], bbox_labels_pred_valid[:, 4:-1], pred[0][:, -1:]
    >>> bboxes_preds_decode = anchors_func.decode_bboxes(bboxes_preds, anchors)
    >>> # Show matched predictions
    >>> data.show_image_with_bboxes(img, bboxes_preds_decode, labels_preds.numpy().argmax(-1), labels_preds.numpy().max(-1))
    >>> # Show gathered bbox ground truth
    >>> data.show_image_with_bboxes(img, bboxes_true, labels_true.numpy().argmax(-1), labels_true.numpy().max(-1))
    >>> # Show gathered encoded, bbox ground truth
    >>> bboxes_true_decode = anchors_func.decode_bboxes(bboxes_true_encoded, anchors)
    >>> data.show_image_with_bboxes(img, bboxes_true_decode, labels_true.numpy().argmax(-1), labels_true.numpy().max(-1))
    """

    def __init__(self, input_shape, pyramid_levels=[3, 5], center_radius=2.5, topk_ious_max=10, grid_zero_start=True, epsilon=1e-8):
        self.center_radius, self.topk_ious_max, self.epsilon = center_radius, topk_ious_max, epsilon
        self.input_shape, self.grid_zero_start = input_shape, grid_zero_start
        # pyramid_levels = get_pyramid_levels_by_num_anchors(self.input_shape, self.num_anchors)
        self.anchors = get_anchors(self.input_shape, pyramid_levels, aspect_ratios=[1], num_scales=1, anchor_scale=1, grid_zero_start=self.grid_zero_start)

        # Anchors constant values
        self.anchors_centers = (self.anchors[:, :2] + self.anchors[:, 2:]) * 0.5
        self.anchors_hws = self.anchors[:, 2:] - self.anchors[:, :2]
        self.anchors_nd = tf.expand_dims(self.anchors, 0)  # [1, num_anchors, 4]
        self.anchors_centers_nd, self.anchors_hws_nd = tf.expand_dims(self.anchors_centers, 0), tf.expand_dims(self.anchors_hws, 0)  # [1, num_anchors, 2]
        self.centers_enlarge_nd = self.anchors_hws_nd * self.center_radius

    def __picking_anchors_by_center_within_bboxes__(self, bboxes_true_nd):
        # get_in_boxes_info https://github.com/Megvii-BaseDetection/YOLOX/tree/master/yolox/models/yolo_head.py#L522
        # bboxes: [[top, left, bottom, right]], anchors: [[top, left, bottom, right]]
        # anchors_centers_nd: [1, num_anchors, 2], bboxes_true_nd: [num_bboxes, 1, 4]
        # is_anchor_in_bbox: [num_bboxes, num_anchors, 2]
        is_anchor_in_bbox = tf.logical_and(bboxes_true_nd[:, :, :2] < self.anchors_centers_nd, bboxes_true_nd[:, :, 2:] > self.anchors_centers_nd)
        is_anchor_in_bbox = tf.reduce_all(is_anchor_in_bbox, axis=-1)  # All 4 points matching: [num_bboxes, num_anchors]

        bboxes_centers_nd = (bboxes_true_nd[:, :, :2] + bboxes_true_nd[:, :, 2:]) * 0.5
        is_anchor_in_center_top_left = self.anchors_centers_nd > (bboxes_centers_nd - self.centers_enlarge_nd)
        is_anchor_in_center_bottom_right = self.anchors_centers_nd < (bboxes_centers_nd + self.centers_enlarge_nd)
        is_anchor_in_center = tf.logical_and(is_anchor_in_center_top_left, is_anchor_in_center_bottom_right)
        is_anchor_in_center = tf.reduce_all(is_anchor_in_center, axis=-1)
        return is_anchor_in_bbox, is_anchor_in_center

    def __decode_bboxes__(self, bboxes_pred, anchors_centers, anchors_hws):
        bboxes_pred_center = bboxes_pred[:, :2] * anchors_hws + anchors_centers
        bboxes_pred_hw = tf.math.exp(bboxes_pred[:, 2:]) * anchors_hws
        bboxes_pred_top_left = bboxes_pred_center - 0.5 * bboxes_pred_hw
        bboxes_pred_bottom_right = bboxes_pred_top_left + bboxes_pred_hw
        return bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_center, bboxes_pred_hw

    def __encode_bboxes__(self, bboxes_true, anchors_centers, anchors_hws):
        # bboxes_true_center, bboxes_true_hw = corners_to_center_yxhw_nd(bboxes_true)
        bboxes_true_hw = bboxes_true[:, 2:] - bboxes_true[:, :2]
        bboxes_true_center = (bboxes_true[:, 2:] + bboxes_true[:, :2]) / 2.0
        bboxes_true_center_encoded = (bboxes_true_center - anchors_centers) / anchors_hws
        bboxes_true_hw_encoded = tf.math.log(bboxes_true_hw / anchors_hws + self.epsilon)
        return tf.concat([bboxes_true_center_encoded, bboxes_true_hw_encoded], axis=-1)

    def __center_iou_nd__(self, bboxes_true_nd, bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_hw):
        # bboxes_true_nd: [num_bboxes, 1, *[top, left, bottom, right]]
        bboxes_pred_top_left_nd, bboxes_pred_bottom_right_nd = tf.expand_dims(bboxes_pred_top_left, 0), tf.expand_dims(bboxes_pred_bottom_right, 0)
        inter_top_left = tf.maximum(bboxes_pred_top_left_nd, bboxes_true_nd[:, :, :2])
        inter_bottom_right = tf.minimum(bboxes_pred_bottom_right_nd, bboxes_true_nd[:, :, 2:])
        inter_hw = tf.maximum(inter_bottom_right - inter_top_left, 0)
        inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]

        bboxes_area_nd = (bboxes_true_nd[:, :, 2] - bboxes_true_nd[:, :, 0]) * (bboxes_true_nd[:, :, 3] - bboxes_true_nd[:, :, 1])
        pred_area_nd = tf.expand_dims(bboxes_pred_hw[:, 0] * bboxes_pred_hw[:, 1], 0)
        union_area = bboxes_area_nd + pred_area_nd - inter_area
        return inter_area / union_area

    def __filter_anchors_matching_multi_bboxes__(self, topk_anchors, cost, check_cond):
        cond = tf.where(check_cond)
        conflict_costs = tf.gather(cost, cond[:, 0], axis=1)
        topk_anchors = tf.tensor_scatter_nd_update(tf.transpose(topk_anchors), cond, tf.zeros([tf.shape(cond)[0], tf.shape(topk_anchors)[0]]))
        topk_anchors = tf.transpose(topk_anchors)
        update_pos = tf.concat([tf.expand_dims(tf.argmin(conflict_costs, axis=0), 1), cond], axis=-1)  # Not considering if argmin cost is in conflict_costs
        topk_anchors = tf.tensor_scatter_nd_update(topk_anchors, update_pos, tf.ones(tf.shape(cond)[0]))
        return topk_anchors

    def __dynamic_k_matching__(self, ious, cost):
        # dynamic_k_matching https://github.com/Megvii-BaseDetection/YOLOX/tree/master/yolox/models/yolo_head.py#L607
        top_ious = tf.sort(ious, direction="DESCENDING")[:, : self.topk_ious_max]
        dynamic_ks = tf.maximum(tf.reduce_sum(top_ious, axis=-1), 1.0)  # [???] why sum up ious

        num_picked = tf.shape(cost)[-1]
        cc = tf.concat([cost, tf.expand_dims(dynamic_ks, 1)], axis=-1)
        # matching_matrix, tf.argsort default direction = "ASCENDING"
        topk_anchors = tf.map_fn(lambda xx: tf.reduce_sum(tf.one_hot(tf.argsort(xx[:-1])[: tf.cast(xx[-1], tf.int32)], num_picked), 0), cc)
        check_cond = tf.reduce_sum(topk_anchors, axis=0) > 1
        return tf.cond(
            tf.reduce_any(check_cond),
            lambda: self.__filter_anchors_matching_multi_bboxes__(topk_anchors, cost, check_cond),
            lambda: topk_anchors,
        )

    def __call__(self, bbox_labels_true, bbox_labels_pred):
        from tensorflow.keras import backend as K

        # get_assignments https://github.com/Megvii-BaseDetection/YOLOX/tree/master/yolox/models/yolo_head.py#425
        bbox_labels_true = tf.gather_nd(bbox_labels_true, tf.where(bbox_labels_true[:, -1] > 0))

        bboxes_true, labels_true = bbox_labels_true[:, :4], bbox_labels_true[:, 4:-1]
        bboxes_true_nd = tf.expand_dims(bboxes_true, 1)

        # is_anchor_in_bbox, is_anchor_in_center: [num_bboxes, num_anchors]
        is_anchor_in_bbox, is_anchor_in_center = self.__picking_anchors_by_center_within_bboxes__(bboxes_true_nd)
        # [num_anchors]
        is_anchor_match_any_bbox = tf.logical_or(tf.reduce_any(is_anchor_in_bbox, axis=0), tf.reduce_any(is_anchor_in_center, axis=0))
        pick_cond = tf.where(is_anchor_match_any_bbox)[:, 0]
        # [num_bboxes, num_picked_anchors]
        is_anchor_valid = tf.logical_and(tf.gather(is_anchor_in_bbox, pick_cond, axis=-1), tf.gather(is_anchor_in_center, pick_cond, axis=-1))

        bbox_labels_pred = bbox_labels_pred[is_anchor_match_any_bbox]
        bboxes_pred, labels_pred, object_pred = bbox_labels_pred[:, :4], bbox_labels_pred[:, 4:-1], bbox_labels_pred[:, -1:]

        # decode_bboxes
        anchors_centers, anchors_hws = self.anchors_centers[is_anchor_match_any_bbox], self.anchors_hws[is_anchor_match_any_bbox]
        bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_center, bboxes_pred_hw = self.__decode_bboxes__(bboxes_pred, anchors_centers, anchors_hws)

        ious = self.__center_iou_nd__(bboxes_true_nd, bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_hw)  # [num_bboxes, num_picked_anchors]
        ious_loss = -tf.math.log(ious + self.epsilon)

        obj_labels_pred = tf.sqrt(labels_pred * object_pred)
        cls_loss = K.binary_crossentropy(tf.expand_dims(labels_true, 1), tf.expand_dims(obj_labels_pred, 0))  # [num_bboxes, num_picked_anchors, num_classes]
        cls_loss = tf.reduce_sum(cls_loss, -1)  # [num_bboxes, num_picked_anchors]
        cost = cls_loss + 3.0 * ious_loss + 1e5 * tf.cast(tf.logical_not(is_anchor_valid), cls_loss.dtype)  # [num_bboxes, num_picked_anchors]

        # dynamic_k_matching
        bbox_matched_k_anchors = self.__dynamic_k_matching__(ious, cost)  # [num_bboxes, num_picked_anchors], contains only 0, 1
        is_anchor_iou_match_any = tf.reduce_any(bbox_matched_k_anchors > 0, axis=0)  # [num_picked_anchors]
        is_anchor_iou_match_any_idx = tf.where(is_anchor_iou_match_any)

        # TODO: is_anchor_iou_match_any_idx.shape[0] == 0
        anchor_best_matching_bbox = tf.argmax(tf.gather(bbox_matched_k_anchors, is_anchor_iou_match_any_idx[:, 0], axis=-1), 0)
        anchor_labels = tf.gather(labels_true, anchor_best_matching_bbox)  # [num_picked_anchors， num_classes]
        pred_iou_loss = tf.reduce_sum(bbox_matched_k_anchors * ious, 0)[is_anchor_iou_match_any]

        # get_losses after get_assignments. Bboxes for iou loss, [top, left, bottom, right]
        out_bboxes_true = tf.gather(bboxes_true, anchor_best_matching_bbox)
        out_labels_true = anchor_labels * tf.expand_dims(pred_iou_loss, -1)

        # object loss, [num_anchors]
        out_object_true = tf.tensor_scatter_nd_update(is_anchor_match_any_bbox, tf.where(is_anchor_match_any_bbox), is_anchor_iou_match_any)
        object_true_idx = tf.where(out_object_true)  # [num_picked_anchors, 1]

        # l1_target loss, encoded [center_top, center_left, height, width]
        anchors_centers_valid = tf.gather_nd(anchors_centers, is_anchor_iou_match_any_idx)
        anchors_hws_valid = tf.gather_nd(anchors_hws, is_anchor_iou_match_any_idx)
        out_bboxes_true_encoded = self.__encode_bboxes__(out_bboxes_true, anchors_centers_valid, anchors_hws_valid)

        # tf.stop_gradient requires returning value been a single tensor with same dtype as inputs.
        return tf.concat([out_bboxes_true, out_bboxes_true_encoded, out_labels_true, tf.cast(object_true_idx, out_bboxes_true.dtype)], axis=-1)
