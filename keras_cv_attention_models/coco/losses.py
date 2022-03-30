import math
import tensorflow as tf
from tensorflow.keras import backend as K


def __bbox_iou__(true_top_left, true_bottom_right, true_hw, pred_top_left, pred_bottom_right, pred_hw, use_ciou=False, epsilon=1e-8):
    # Use all top_left, bottom_right, hw as parameters, as hw usaully already calculated before calling this function, like center_hw bboxes.
    inter_top_left = tf.maximum(true_top_left, pred_top_left)
    inter_bottom_right = tf.minimum(true_bottom_right, pred_bottom_right)
    inter_hw = tf.maximum(inter_bottom_right - inter_top_left, 0)
    inter_area = inter_hw[:, 0] * inter_hw[:, 1]

    bboxes_trues_area = true_hw[:, 0] * true_hw[:, 1]
    bboxes_preds_area = pred_hw[:, 0] * pred_hw[:, 1]
    union_area = bboxes_trues_area + bboxes_preds_area - inter_area
    iou = inter_area / (union_area + epsilon)
    if use_ciou:
        # https://github.com/WongKinYiu/yolor/blob/main/utils/general.py#L187
        outer_top_left = tf.minimum(true_top_left, pred_top_left)
        outer_bottom_right = tf.maximum(true_bottom_right, pred_bottom_right)
        outer_hw = outer_bottom_right - outer_top_left
        outer_area = outer_hw[:, 0] ** 2 + outer_hw[:, 1] ** 2 + epsilon

        rho_height = (true_top_left[:, 0] + true_bottom_right[:, 0] - pred_top_left[:, 0] - pred_bottom_right[:, 0]) ** 2
        rho_width = (true_top_left[:, 1] + true_bottom_right[:, 1] - pred_top_left[:, 1] - pred_bottom_right[:, 1]) ** 2
        rho = (rho_height + rho_width) / 4
        vv_scale = 4 / math.pi ** 2
        vv = vv_scale * (tf.atan(true_hw[:, 1] / (true_hw[:, 0] + epsilon)) - tf.atan(pred_hw[:, 1] / (pred_hw[:, 0] + epsilon))) ** 2
        alpha = tf.stop_gradient(vv / ((1 + epsilon) - iou + vv))
        return iou - (rho / outer_area + vv * alpha)
    else:
        return iou


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class FocalLossWithBbox(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=1.5, delta=0.1, bbox_loss_weight=50.0, label_smoothing=0.0, from_logits=False, **kwargs):
        # https://github.com/google/automl/tree/master/efficientdet/hparams_config.py#L229
        # classification loss: alpha, gamma, label_smoothing = 0.25, 1.5, 0.0
        # localization loss: delta, box_loss_weight = 0.1, 50.0
        super().__init__(**kwargs)
        self.alpha, self.gamma, self.delta, self.bbox_loss_weight = alpha, gamma, delta, bbox_loss_weight
        self.label_smoothing, self.from_logits = label_smoothing, from_logits
        # self.huber = tf.keras.losses.Huber(self.delta, reduction=tf.keras.losses.Reduction.NONE)

        self.class_acc = tf.Variable(0, dtype="float32", trainable=False)
        # self.class_acc = tf.Variable(0, dtype="float32", trainable=False, aggregation=tf.VariableAggregation.MEAN)

    def __focal_loss__(self, class_true_valid, class_pred_valid):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/train_lib.py#L257
        if self.from_logits:
            class_pred_valid = tf.sigmoid(class_pred_valid)
        # 1 -> 0.25, 0 -> 0.75
        # alpha_factor = class_true_valid * self.alpha + (1 - class_true_valid) * (1 - self.alpha)
        cond = tf.equal(class_true_valid, 1.0)
        alpha_factor = tf.where(cond, self.alpha, (1.0 - self.alpha))
        # p_t = class_true_valid * class_pred_valid + (1 - class_true_valid) * (1 - class_pred_valid)
        p_t = tf.where(cond, class_pred_valid, (1.0 - class_pred_valid))
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        if self.label_smoothing > 0:
            class_true_valid = class_true_valid * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        # focal_factor = (1 - output) ** gamma if class is 1 else output ** gamma
        # focal_bce = K.binary_focal_crossentropy(class_true_valid, class_pred_valid, gamma=self.gamma, from_logits=True)
        # focal_bce = focal_factor * K.binary_crossentropy(class_true_valid, class_pred_valid)
        # ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=class_true_valid, logits=class_pred_valid)
        ce = K.binary_crossentropy(class_true_valid, class_pred_valid)
        return alpha_factor * focal_factor * ce

    def __bbox_loss__(self, bbox_true_valid, bbox_pred_valid):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/train_lib.py#L409
        # error = tf.subtract(bbox_pred_valid, bbox_true_valid)
        # abs_error = tf.abs(error)
        # regression_loss = tf.where(abs_error <= self.delta, 0.5 * tf.square(error), self.delta * abs_error - 0.5 * tf.square(self.delta))
        # regression_loss / self.delta -> torch one
        # regression_loss = tf.where(abs_error <= self.delta, 0.5 * (abs_error ** 2) / self.delta, abs_error - 0.5 * self.delta)
        # tf.losses.huber <--> tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.AUTO)
        vv = tf.cond(
            tf.shape(bbox_true_valid)[0] == 0,
            lambda: 0.0,
            lambda: tf.losses.huber(bbox_true_valid, bbox_pred_valid, self.delta),
            # lambda: self.huber(tf.expand_dims(bbox_true_valid, -1), tf.expand_dims(bbox_pred_valid, -1)) / 4.0,
        )
        return tf.cast(vv, bbox_pred_valid.dtype)

    def call(self, y_true, y_pred):
        bbox_pred, class_pred = y_pred[:, :, :4], y_pred[:, :, 4:]
        bbox_true, class_true, anchor_mark = y_true[:, :, :4], y_true[:, :, 4:-1], y_true[:, :, -1]
        exclude_ignored_pick = tf.where(anchor_mark != -1)
        valid_pick = tf.where(anchor_mark == 1)
        num_positive_anchors = tf.cast(tf.maximum(tf.shape(valid_pick)[0], 1), y_pred.dtype)

        # tf.gather_nd works better than tf.gather
        class_true_valid, class_pred_valid = tf.gather_nd(class_true, exclude_ignored_pick), tf.gather_nd(class_pred, exclude_ignored_pick)
        bbox_true_valid, bbox_pred_valid = tf.gather_nd(bbox_true, valid_pick), tf.gather_nd(bbox_pred, valid_pick)

        cls_loss = self.__focal_loss__(class_true_valid, class_pred_valid)  # divide before sum, if meet inf
        bbox_loss = self.__bbox_loss__(bbox_true_valid, bbox_pred_valid)
        cls_loss, bbox_loss = tf.reduce_sum(cls_loss) / num_positive_anchors, tf.reduce_sum(bbox_loss) / num_positive_anchors

        # Calulate accuracy here, will use it in metrics
        self.class_acc.assign(tf.reduce_mean(tf.cast(tf.argmax(class_pred_valid, axis=-1) == tf.argmax(class_true_valid, axis=-1), "float32")))

        # return bbox_loss
        tf.print(" - cls_loss:", cls_loss, "- bbox_loss:", bbox_loss, end="\r")
        return cls_loss + bbox_loss * self.bbox_loss_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "delta": self.delta,
                "bbox_loss_weight": self.bbox_loss_weight,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class AnchorFreeLoss(tf.keras.losses.Loss):
    """
    # Basic test:
    >>> from keras_cv_attention_models.coco import losses, anchors_func
    >>> aa = losses.AnchorFreeLoss(input_shape=(640, 640), use_l1_loss=True)

    >>> from keras_cv_attention_models import yolox, test_images
    >>> from keras_cv_attention_models.coco import anchors_func, data
    >>> mm = yolox.YOLOXS()
    >>> img = test_images.dog_cat()
    >>> pred = mm(mm.preprocess_input(img))

    >>> bbs, lls, ccs = mm.decode_predictions(pred)[0]
    >>> bbox_labels_true = tf.concat([bbs, tf.one_hot(lls, 80), tf.ones([bbs.shape[0], 1])], axis=-1)
    >>> print("\n", aa(tf.expand_dims(bbox_labels_true, 0), pred))
    >>> # - l1_loss: 0.323877126 - cls_loss: 0.416673392 - bbox_loss: 0.129255757 - obj_loss: 0.961375535
    >>> # tf.Tensor(2.3482049, shape=(), dtype=float32)

    # Test with dataset:
    >>> from keras_cv_attention_models import coco, yolox
    >>> train_dataset = coco.init_dataset(batch_size=8, use_anchor_free_mode=True, anchor_pyramid_levels=[3, 5], rescale_mode="raw")[0]
    >>> images, bboxes_labels = train_dataset.as_numpy_iterator().next()
    >>> mm = yolox.YOLOXS(input_shape=(256, 256, 3))
    >>> preds = mm(images)
    >>> loss = coco.losses.AnchorFreeLoss(mm.input_shape[1:-1])
    >>> print(f"\n{loss(bboxes_labels, preds) = }")
    >>> # - cls_loss: 1.16732407 - bbox_loss: 0.53272897 - obj_loss: 2.27287531
    >>> # loss(bboxes_labels, preds) = <tf.Tensor: shape=(), dtype=float32, numpy=6.103844>
    """

    def __init__(
        self,
        input_shape,  # Required for initing anchors...
        pyramid_levels=[3, 5],  # Required for initing anchors...
        use_l1_loss=False,
        bbox_loss_weight=5.0,
        anchor_assign_center_radius=2.5,
        anchor_assign_topk_ious_max=10,
        anchor_grid_zero_start=True,
        epsilon=1e-8,
        label_smoothing=0.0,
        from_logits=False,
        **kwargs
    ):
        from keras_cv_attention_models.coco import anchors_func

        super().__init__(**kwargs)
        self.bbox_loss_weight, self.use_l1_loss, self.epsilon = bbox_loss_weight, use_l1_loss, epsilon
        self.label_smoothing, self.from_logits = label_smoothing, from_logits
        self.input_shape, self.pyramid_levels, self.anchor_grid_zero_start = input_shape, pyramid_levels, anchor_grid_zero_start
        self.anchor_assign_center_radius, self.anchor_assign_topk_ious_max = anchor_assign_center_radius, anchor_assign_topk_ious_max
        self.anchor_assign = anchors_func.AnchorFreeAssignMatching(
            input_shape, pyramid_levels, anchor_assign_center_radius, anchor_assign_topk_ious_max, anchor_grid_zero_start, epsilon=epsilon
        )
        self.class_acc = tf.Variable(0, dtype="float32", trainable=False)
        # self.class_acc = tf.Variable(0, dtype="float32", trainable=False, aggregation=tf.VariableAggregation.MEAN)

    def __iou_loss__(self, bboxes_trues, pred_top_left, pred_bottom_right, pred_hw):
        # bboxes_trues: [[top, left, bottom, right]]
        # inter_top_left = tf.maximum(bboxes_trues[:, :2], pred_top_left)
        # inter_bottom_right = tf.minimum(bboxes_trues[:, 2:], pred_bottom_right)
        # inter_hw = tf.maximum(inter_bottom_right - inter_top_left, 0)
        # inter_area = inter_hw[:, 0] * inter_hw[:, 1]
        #
        # bboxes_trues_area = (bboxes_trues[:, 2] - bboxes_trues[:, 0]) * (bboxes_trues[:, 3] - bboxes_trues[:, 1])
        # bboxes_preds_area = pred_hw[:, 0] * pred_hw[:, 1]
        # union_area = bboxes_trues_area + bboxes_preds_area - inter_area
        # iou = inter_area / (union_area + self.epsilon)
        true_top_left, true_bottom_right = bboxes_trues[:, :2], bboxes_trues[:, 2:4]
        true_hw = true_bottom_right - true_top_left
        iou = __bbox_iou__(true_top_left, true_bottom_right, true_hw, pred_top_left, pred_bottom_right, pred_hw, epsilon=self.epsilon)
        return 1 - iou ** 2

    def __valid_call_single__(self, bbox_labels_true, bbox_labels_pred):
        bbox_labels_true_assined = tf.stop_gradient(self.anchor_assign(bbox_labels_true, bbox_labels_pred))
        bboxes_true, bboxes_true_encoded, labels_true, object_true_idx_nd = tf.split(bbox_labels_true_assined, [4, 4, -1, 1], axis=-1)
        object_true_idx_nd = tf.cast(object_true_idx_nd, tf.int32)
        object_true = tf.tensor_scatter_nd_update(tf.zeros_like(bbox_labels_pred[:, -1]), object_true_idx_nd, tf.ones_like(bboxes_true[:, -1]))

        # object_true_idx = object_true_idx_nd[:, 0]
        # bbox_labels_pred_valid = tf.gather(bbox_labels_pred, object_true_idx)
        bbox_labels_pred_valid = tf.gather_nd(bbox_labels_pred, object_true_idx_nd)
        bboxes_pred, labels_pred, object_pred = bbox_labels_pred_valid[:, :4], bbox_labels_pred_valid[:, 4:-1], bbox_labels_pred[:, -1]
        # bboxes_true.set_shape(bboxes_pred.shape)

        # anchors_centers = tf.gather(self.anchor_assign.anchors_centers, object_true_idx)
        # anchors_hws = tf.gather(self.anchor_assign.anchors_hws, object_true_idx)
        anchors_centers = tf.gather_nd(self.anchor_assign.anchors_centers, object_true_idx_nd)
        anchors_hws = tf.gather_nd(self.anchor_assign.anchors_hws, object_true_idx_nd)
        bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_center, bboxes_pred_hw = self.anchor_assign.__decode_bboxes__(
            bboxes_pred, anchors_centers, anchors_hws
        )

        if self.label_smoothing > 0:
            labels_true = labels_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        class_loss = tf.reduce_sum(K.binary_crossentropy(labels_true, labels_pred))
        object_loss = tf.reduce_sum(K.binary_crossentropy(tf.cast(object_true, object_pred.dtype), object_pred))
        bbox_loss = tf.reduce_sum(self.__iou_loss__(bboxes_true, bboxes_pred_top_left, bboxes_pred_bottom_right, bboxes_pred_hw))
        if self.use_l1_loss:
            l1_loss = tf.reduce_sum(tf.abs(bboxes_true_encoded - bboxes_pred))  # mean absolute error
        else:
            l1_loss = 0.0

        num_valid_anchors = tf.cast(tf.shape(bboxes_pred)[0], bboxes_pred.dtype)
        class_acc = tf.reduce_mean(tf.cast(tf.argmax(labels_true, axis=-1) == tf.argmax(labels_pred, axis=-1), "float32"))
        return class_loss, bbox_loss, object_loss, l1_loss, num_valid_anchors, class_acc

    def __call_single__(self, inputs):
        bbox_labels_true, bbox_labels_pred = inputs[0], inputs[1]
        return tf.cond(
            tf.reduce_any(bbox_labels_true[:, -1] > 0),
            lambda: self.__valid_call_single__(bbox_labels_true, bbox_labels_pred),
            lambda: (0.0, 0.0, tf.reduce_sum(K.binary_crossentropy(0.0, bbox_labels_pred[:, -1])), 0.0, 0.0, 0.0),  # Object loss only, target is all False.
        )

    def call(self, y_true, y_pred):
        if self.from_logits:
            bbox_pred, class_pred = y_pred[:, :, :4], y_pred[:, :, 4:]
            class_pred = tf.sigmoid(class_pred)
            y_pred = tf.concat([bbox_pred, class_pred], axis=-1)

        out_dtype = (y_pred.dtype,) * 6
        class_loss, bbox_loss, object_loss, l1_loss, num_valid, class_acc = tf.map_fn(self.__call_single__, (y_true, y_pred), fn_output_signature=out_dtype)

        num_valid = tf.maximum(tf.reduce_sum(num_valid), 1.0)
        class_loss, bbox_loss, l1_loss = tf.reduce_sum(class_loss) / num_valid, tf.reduce_sum(bbox_loss) / num_valid, tf.reduce_sum(l1_loss) / num_valid
        object_loss = tf.reduce_sum(object_loss) / num_valid  # [ ??? ] why not divide actual object shape?

        # Calulate accuracy here, will use it in metrics
        self.class_acc.assign(tf.reduce_mean(class_acc))

        if self.use_l1_loss:
            tf.print(" - l1_loss:", l1_loss, end="")
        tf.print(" - cls_loss:", class_loss, "- bbox_loss:", bbox_loss, "- obj_loss:", object_loss, end="\r")
        return class_loss + object_loss + l1_loss + bbox_loss * self.bbox_loss_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "pyramid_levels": self.pyramid_levels,
                "use_l1_loss": self.use_l1_loss,
                "bbox_loss_weight": self.bbox_loss_weight,
                "anchor_assign_center_radius": self.anchor_assign_center_radius,
                "anchor_assign_topk_ious_max": self.anchor_assign_topk_ious_max,
                "anchor_grid_zero_start": self.anchor_grid_zero_start,
                "epsilon": self.epsilon,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class YOLORLossWithBbox(tf.keras.losses.Loss):
    """
    # Test with dataset:
    >>> from keras_cv_attention_models import coco, yolor
    >>> train_dataset = coco.init_dataset(batch_size=8, use_yolor_anchors_mode=True, anchor_pyramid_levels=[3, 5], rescale_mode="raw01")[0]
    >>> images, bboxes_labels = train_dataset.as_numpy_iterator().next()
    >>> mm = yolor.YOLOR_CSP(input_shape=(256, 256, 3))
    >>> preds = mm(images)
    >>> loss = coco.losses.YOLORLossWithBbox(mm.input_shape[1:-1])
    >>> print(f"\n{loss(bboxes_labels, preds) = }")
    >>> # - cls_loss: 0.014720859 - bbox_loss: 0.167933762 - obj_loss: 0.0272860527
    >>> # loss(bboxes_labels, preds) = <tf.Tensor: shape=(), dtype=float32, numpy=0.8959118>
    """

    def __init__(
        self, input_shape, pyramid_levels=[3, 5], gamma=0.0, class_loss_weight=0.5, bbox_loss_weight=0.05, label_smoothing=0.0, from_logits=False, **kwargs
    ):
        from keras_cv_attention_models.coco import anchors_func

        num_pyramid_levels = max(pyramid_levels) - min(pyramid_levels) + 1
        self.loss_scale = 3.0 / tf.cast(num_pyramid_levels, "float32")
        anchors = anchors_func.get_yolor_anchors(input_shape[:2], pyramid_levels, is_for_training=False)
        anchors = anchors[:, 2:4] / anchors[:, 4:]  # Anchors is divided with strides in YOLOLayer
        self.anchors = tf.expand_dims(anchors, 0)  # [1, total_anchors, 2], first dimension is batch, last dimension is anchor_ratio
        feature_sizes = anchors_func.get_feature_sizes(input_shape, pyramid_levels)[min(pyramid_levels) : max(pyramid_levels) + 1]
        OBJECT_LEVEL_WEIGHTS = {3: [4.0, 1.0, 0.4], 4: [4.0, 1.0, 0.4, 0.1], 5: [4.0, 1.0, 0.5, 0.4, 0.1]}  # balance in original yolor loss
        object_level_weights = []
        for feature_size, object_level_weight in zip(feature_sizes, OBJECT_LEVEL_WEIGHTS.get(num_pyramid_levels, OBJECT_LEVEL_WEIGHTS[5])):
            object_level_weights.extend([object_level_weight] * (feature_size[0] * feature_size[1] * 3))
        self.object_level_weights = tf.convert_to_tensor(object_level_weights) * (1.4 if num_pyramid_levels >= 4 else 1.0)

        super().__init__(**kwargs)
        self.input_shape, self.pyramid_levels = input_shape, pyramid_levels
        self.gamma, self.class_loss_weight, self.bbox_loss_weight = gamma, class_loss_weight, bbox_loss_weight
        self.label_smoothing, self.from_logits = label_smoothing, from_logits

        self.class_acc = tf.Variable(0, dtype="float32", trainable=False)

    def __bbox_decode_and_ciou__(self, y_true_valid, y_pred_valid, anchors):
        # compute_loss https://github.com/WongKinYiu/yolor/blob/main/utils/loss.py#L62
        true_center, true_hw = y_true_valid[:, :2], y_true_valid[:, 2:4]
        true_top_left = true_center - true_hw / 2
        true_bottom_right = true_top_left + true_hw

        y_pred_valid_centers = y_pred_valid[:, :2] * 2 - 0.5
        y_pred_valid_hw = (y_pred_valid[:, 2:4] * 2) ** 2 * anchors
        pred_top_left = y_pred_valid_centers - y_pred_valid_hw / 2
        pred_bottom_right = pred_top_left + y_pred_valid_hw
        return __bbox_iou__(true_top_left, true_bottom_right, true_hw, pred_top_left, pred_bottom_right, y_pred_valid_hw, use_ciou=True)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        batch_size = tf.shape(y_true)[0]
        anchor_mark = y_true[:, :, -1]
        valid_pick = tf.where(anchor_mark == 1)
        # num_positive_anchors = tf.cast(tf.maximum(tf.shape(valid_pick)[0], 1), y_pred.dtype)

        # tf.gather_nd works better than tf.gather
        y_true_valid, y_pred_valid = tf.gather_nd(y_true, valid_pick), tf.gather_nd(y_pred, valid_pick)
        anchors = tf.gather_nd(tf.repeat(self.anchors, batch_size, axis=0), valid_pick)

        iou = self.__bbox_decode_and_ciou__(y_true_valid, y_pred_valid, anchors)
        bbox_loss = tf.reduce_mean(1 - iou) * self.loss_scale

        # tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio, gr = 1.0
        object_true = tf.tensor_scatter_nd_update(tf.zeros_like(y_pred[:, :, -1]), valid_pick, tf.maximum(iou, 0))
        object_loss = K.binary_crossentropy(object_true, y_pred[:, :, -1])
        # print(object_true.shape, object_loss.shape, self.object_level_weights.shape)
        object_loss = tf.reduce_mean(object_loss * self.object_level_weights) * self.loss_scale

        class_true_valid, class_pred_valid = y_true_valid[:, 4:-1], y_pred_valid[:, 4:-1]
        # if self.label_smoothing > 0:
        #     class_true_valid = class_true_valid * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.gamma > 0:
            class_loss = tf.losses.binary_focal_crossentropy(class_true_valid, class_pred_valid, gamma=self.gamma, label_smoothing=self.label_smoothing)
        else:
            class_loss = tf.losses.binary_crossentropy(class_true_valid, class_pred_valid, label_smoothing=self.label_smoothing)
        class_loss = tf.reduce_mean(class_loss) * self.loss_scale

        # Calulate accuracy here, will use it in metrics
        self.class_acc.assign(tf.reduce_mean(tf.cast(tf.argmax(class_pred_valid, axis=-1) == tf.argmax(class_true_valid, axis=-1), "float32")))

        # return bbox_loss
        tf.print(" - cls_loss:", class_loss, "- bbox_loss:", bbox_loss, "- obj_loss:", object_loss, end="\r")
        return (class_loss * self.class_loss_weight + bbox_loss * self.bbox_loss_weight + object_loss) * tf.cast(batch_size, y_pred.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "pyramid_levels": self.pyramid_levels,
                "gamma": self.gamma,
                "class_loss_weight": self.class_loss_weight,
                "bbox_loss_weight": self.bbox_loss_weight,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class ClassAccuracyWithBbox(tf.keras.metrics.Metric):
    def __init__(self, name="cls_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cls_acc = self.add_weight(name="cls_acc", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pick = tf.where(y_true[:, :, -1] > 0)
        cls_true_valid = tf.argmax(tf.gather_nd(y_true[:, :, 4:-1], pick), axis=-1)
        cls_pred_valid = tf.argmax(tf.gather_nd(y_pred[:, :, 4:], pick), axis=-1)
        cls_acc = tf.reduce_mean(tf.cast(cls_true_valid == cls_pred_valid, "float32"))
        # tf.assert_less(cls_acc, 1.1)
        self.cls_acc.assign_add(self.loss_calss_with_acc.cls_acc)
        self.count.assign_add(1.0)


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class ClassAccuracyWithBboxWrapper(tf.keras.metrics.Metric):
    def __init__(self, loss_class_with_acc=None, name="cls_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_acc = self.add_weight(name="cls_acc", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")
        # self.class_acc = self.add_weight(name="cls_acc", initializer="zeros", dtype="float32", aggregation=tf.VariableAggregation.MEAN)
        # self.count = self.add_weight(name="count", initializer="zeros", dtype="float32", aggregation=tf.VariableAggregation.MEAN)
        self.loss_class_with_acc = loss_class_with_acc

    def update_state(self, y_true=None, y_pred=None, sample_weight=None):
        self.class_acc.assign_add(self.loss_class_with_acc.class_acc)
        self.count.assign_add(1.0)

    def result(self):
        return self.class_acc / self.count
