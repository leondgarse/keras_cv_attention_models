import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv_attention_models.coco import anchors_func, data
from tqdm import tqdm


class DecodePredictions:
    """
    The most simple version decoding prediction and NMS:

    >>> from keras_cv_attention_models import efficientdet, test_images
    >>> model = efficientdet.EfficientDetD0()
    >>> preds = model(model.preprocess_input(test_images.dog()))

    # Decode and NMS
    >>> from keras_cv_attention_models import coco
    >>> input_shape = model.input_shape[1:-1]
    >>> anchors = coco.get_anchors(input_shape=input_shape, pyramid_levels=[3, 7], anchor_scale=4)
    >>> dd = coco.decode_bboxes(preds[0], anchors).numpy()
    >>> rr = tf.image.non_max_suppression(dd[:, :4], dd[:, 4:].max(-1), score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
    >>> dd_nms = tf.gather(dd, rr).numpy()
    >>> bboxes, labels, scores = dd_nms[:, :4], dd_nms[:, 4:].argmax(-1), dd_nms[:, 4:].max(-1)
    >>> print(f"{bboxes = }, {labels = }, {scores = }")
    >>> # bboxes = array([[0.433231  , 0.54432285, 0.8778939 , 0.8187578 ]], dtype=float32), labels = array([17]), scores = array([0.85373735], dtype=float32)
    """

    def __init__(
        self, input_shape=512, pyramid_levels=[3, 7], anchor_scale=4, use_anchor_free_mode=False, use_yolor_anchors_mode=False, use_object_scores="auto"
    ):
        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        self.use_object_scores = (use_yolor_anchors_mode or use_anchor_free_mode) if use_object_scores == "auto" else use_object_scores
        self.anchor_scale, self.use_anchor_free_mode, self.use_yolor_anchors_mode = anchor_scale, use_anchor_free_mode, use_yolor_anchors_mode
        if input_shape is not None and (isinstance(input_shape, (list, tuple)) and input_shape[0] is not None):
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None

    def __init_anchor__(self, input_shape):
        input_shape = input_shape[:2] if isinstance(input_shape, (list, tuple)) else (input_shape, input_shape)
        if self.use_anchor_free_mode:
            self.anchors = anchors_func.get_anchor_free_anchors(input_shape, self.pyramid_levels)
        elif self.use_yolor_anchors_mode:
            self.anchors = anchors_func.get_yolor_anchors(input_shape, self.pyramid_levels)
        else:
            aspect_ratios, num_scales, grid_zero_start = [1, 2, 0.5], 3, False
            self.anchors = anchors_func.get_anchors(input_shape, self.pyramid_levels, aspect_ratios, num_scales, self.anchor_scale, grid_zero_start)
        return self.anchors

    def __topk_class_boxes_single__(self, pred, topk=5000):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L82
        bbox_outputs, class_outputs = pred[:, :4], pred[:, 4:]
        num_classes = class_outputs.shape[-1]
        class_outputs_flatten = tf.reshape(class_outputs, -1)
        _, class_topk_indices = tf.nn.top_k(class_outputs_flatten, k=topk, sorted=False)
        # get original indices for class_outputs, original_indices_hh -> picking indices, original_indices_ww -> picked labels
        original_indices_hh, original_indices_ww = class_topk_indices // num_classes, class_topk_indices % num_classes
        class_indices = tf.stack([original_indices_hh, original_indices_ww], axis=-1)
        scores_topk = tf.gather_nd(class_outputs, class_indices)
        bboxes_topk = tf.gather(bbox_outputs, original_indices_hh)
        return bboxes_topk, scores_topk, original_indices_ww, original_indices_hh

    # def __nms_per_class__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
    #     # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
    #     # Not using, same result with `torchvision.ops.batched_nms`
    #     rrs = []
    #     for ii in tf.unique(labels)[0]:
    #         pick = tf.where(labels == ii)
    #         bb, cc = tf.gather_nd(bbs, pick), tf.gather_nd(ccs, pick)
    #         rr, nms_scores = tf.image.non_max_suppression_with_scores(bb, cc, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
    #         bb_nms = tf.gather(bb, rr)
    #         rrs.append(tf.concat([bb_nms, tf.ones([bb_nms.shape[0], 1]) * tf.cast(ii, bb_nms.dtype), tf.expand_dims(nms_scores, 1)], axis=-1))
    #     rrs = tf.concat(rrs, axis=0)
    #     if tf.shape(rrs)[0] > max_output_size:
    #         score_top_k = tf.argsort(rrs[:, -1], direction="DESCENDING")[:max_output_size]
    #         rrs = tf.gather(rrs, score_top_k)
    #     bboxes, labels, scores = rrs[:, :4], rrs[:, 4], rrs[:, -1]
    #     return bboxes.numpy(), labels.numpy(), scores.numpy()

    def __nms_per_class__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        # From torchvision.ops.batched_nms strategy: in order to perform NMS independently per class. we add an offset to all the boxes.
        # The offset is dependent only on the class idx, and is large enough so that boxes from different classes do not overlap
        # Same result with per_class method: https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
        cls_offset = tf.cast(labels, bbs.dtype) * (tf.reduce_max(bbs) + 1)
        bbs_per_class = bbs + tf.expand_dims(cls_offset, -1)
        rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs_per_class, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, rr).numpy(), tf.gather(labels, rr).numpy(), nms_scores.numpy()

    def __nms_global__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, rr).numpy(), tf.gather(labels, rr).numpy(), nms_scores.numpy()

    def __object_score_split__(self, pred):
        return pred[:, :-1], pred[:, -1]  # May overwrite

    def __decode_single__(self, pred, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="hard", mode="global", topk=-1, input_shape=None):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        if input_shape is not None:
            self.__init_anchor__(input_shape)

        if self.use_object_scores:  # YOLO outputs: [bboxes, classses_score, object_score]
            pred, object_scores = self.__object_score_split__(pred)

        if topk > 0:
            bbs, ccs, labels, picking_indices = self.__topk_class_boxes_single__(pred, topk)
            anchors = tf.gather(self.anchors, picking_indices)
            if self.use_object_scores:
                ccs *= tf.gather(object_scores, picking_indices)
        else:
            bbs, ccs, labels = pred[:, :4], tf.reduce_max(pred[:, 4:], axis=-1), tf.argmax(pred[:, 4:], axis=-1)
            anchors = self.anchors
            if self.use_object_scores:
                ccs *= object_scores

        bbs_decoded = anchors_func.decode_bboxes(bbs, anchors)
        iou_threshold, soft_nms_sigma = (1.0, iou_or_sigma / 2) if method.lower() == "gaussian" else (iou_or_sigma, 0.0)

        if mode == "per_class":
            return self.__nms_per_class__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)
        else:
            return self.__nms_global__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)

    def __call__(self, preds, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="hard", mode="global", topk=-1, input_shape=None):
        """
        https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        iou_or_sigma: means `soft_nms_sigma` if method is "gaussian", else `iou_threshold`.
        method: "gaussian" or "hard".
        mode: "global" or "per_class". "per_class" is strategy from `torchvision.ops.batched_nms`
        topk: `> 0` value for picking topk preds using scores.
        """
        preds = preds if len(preds.shape) == 3 else [preds]
        return [self.__decode_single__(pred, score_threshold, iou_or_sigma, max_output_size, method, mode, topk, input_shape) for pred in preds]


def scale_bboxes_back_single(bboxes, image_shape, scale, target_shape):
    height, width = target_shape[0] / scale, target_shape[1] / scale
    bboxes *= [height, width, height, width]
    bboxes = tf.clip_by_value(bboxes, 0, clip_value_max=[image_shape[0], image_shape[1], image_shape[0], image_shape[1]])
    # [top, left, bottom, right] -> [left, top, width, height]
    bboxes = tf.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]], axis=-1)
    return bboxes


def init_eval_dataset(
    data_name="coco/2017", target_shape=(512, 512), batch_size=8, rescale_mode="torch", resize_method="bilinear", resize_antialias=False, use_bgr_input=False
):
    mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std

    if data_name.endswith(".json"):
        dataset, _, _ = data.detection_dataset_from_custom_json(data_name)
        image_func = lambda image: data.tf_imread(image)
    else:
        dataset = tfds.load(data_name, with_info=False)
        image_func = lambda image: tf.cast(image, tf.float32)
    ds = dataset.get("validation", dataset.get("test", None))
    resize_func = lambda image: data.aspect_aware_resize_and_crop_image(image, target_shape, method=resize_method, antialias=resize_antialias)
    # resize_func = lambda image: data.letterbox_scale_down_only(image, target_shape, method=resize_method, antialias=resize_antialias)
    # ds: [resized_image, scale, original_image_shape, image_id], automl bahavior: rescale -> resize
    ds = ds.map(lambda datapoint: (*resize_func(rescaling(image_func(datapoint["image"]))), tf.shape(datapoint["image"])[:2], datapoint["image/id"]))
    ds = ds.batch(batch_size)
    # ds = ds.map(lambda datapoint: (*resize_func(datapoint["image"]), tf.shape(datapoint["image"])[:2], datapoint["image/id"]))
    # ds = ds.batch(batch_size).map(lambda ii, jj, kk, mm: (rescaling(ii), jj, kk, mm))
    if use_bgr_input:
        ds = ds.map(lambda resized_image, scale, original_image_shape, image_id: (resized_image[:, :, :, ::-1], scale, original_image_shape, image_id))
    return ds


def model_eval_results(
    model, eval_dataset, pred_decoder, nms_score_threshold=0.001, nms_iou_or_sigma=0.5, nms_method="gaussian", nms_mode="per_class", nms_topk=5000
):
    target_shape = (eval_dataset.element_spec[0].shape[1], eval_dataset.element_spec[0].shape[2])
    num_classes = model.output_shape[-1] - 4
    to_91_labels = (lambda label: label + 1) if num_classes >= 90 else (lambda label: data.COCO_80_to_90_LABEL_DICT[label] + 1)
    # Format: [image_id, x, y, width, height, score, class]
    to_coco_eval_single = lambda image_id, bbox, label, score: [image_id, *bbox.tolist(), score, to_91_labels(label)]

    results = []
    for images, scales, original_image_shapes, image_ids in tqdm(eval_dataset):
        preds = model(images)
        preds = tf.cast(preds, tf.float32)
        # decoded_preds: [[bboxes, labels, scores], [bboxes, labels, scores], ...]
        decoded_preds = pred_decoder(preds, nms_score_threshold, iou_or_sigma=nms_iou_or_sigma, method=nms_method, mode=nms_mode, topk=nms_topk)

        for rr, image_shape, scale, image_id in zip(decoded_preds, original_image_shapes, scales, image_ids):  # Loop on batch
            bboxes, labels, scores = rr
            image_id = image_id.numpy()
            bboxes = scale_bboxes_back_single(bboxes, image_shape, scale, target_shape).numpy()
            results.extend([to_coco_eval_single(image_id, bb, cc, ss) for bb, cc, ss in zip(bboxes, labels, scores)])  # Loop on prediction results
    return tf.convert_to_tensor(results).numpy()


def coco_evaluation(detection_results, annotation_file=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if annotation_file is None:
        url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/coco_annotations_instances_val2017.json"
        annotation_file = tf.keras.utils.get_file(origin=url)
    coco_gt = COCO(annotation_file)
    image_ids = list(set(detection_results[:, 0]))
    coco_dt = coco_gt.loadRes(detection_results)
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def run_coco_evaluation(
    model,
    data_name="coco/2017",  # init_eval_dataset parameters
    input_shape=None,
    batch_size=8,
    resize_method="bilinear",
    resize_antialias=False,
    rescale_mode="auto",
    use_bgr_input=False,
    take_samples=-1,
    nms_score_threshold=0.001,  # model_eval_results parameters
    nms_iou_or_sigma=0.5,
    nms_method="gaussian",
    nms_mode="per_class",
    nms_topk=5000,
    use_anchor_free_mode=False,  # model anchors related parameters
    use_yolor_anchors_mode=False,
    anchor_scale=4,  # Init anchors for model prediction. "auto" means 1 if use_anchor_free_mode else 4
    annotation_file=None,
    **anchor_kwargs,
):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    print(">>>> Using input_shape {}.".format(input_shape))

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    eval_dataset = init_eval_dataset(data_name, input_shape, batch_size, rescale_mode, resize_method, resize_antialias, use_bgr_input)
    if take_samples > 0:
        eval_dataset = eval_dataset.take(take_samples)

    if hasattr(model, "decode_predictions") and model.decode_predictions is not None:
        print(">>>> Using model preset decode_predictions")
        pred_decoder = model.decode_predictions
    else:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(input_shape, total_anchors=model.output_shape[1])
        print(">>>> Build decoder, pyramid_levels:", pyramid_levels)
        pred_decoder = DecodePredictions(input_shape, pyramid_levels, anchor_scale, use_anchor_free_mode, use_yolor_anchors_mode)
    detection_results = model_eval_results(model, eval_dataset, pred_decoder, nms_score_threshold, nms_iou_or_sigma, nms_method, nms_mode, nms_topk)
    return coco_evaluation(detection_results, annotation_file)
