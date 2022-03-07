import tensorflow as tf
from tensorflow.keras import backend as K


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

        class_true_valid, class_pred_valid = tf.gather_nd(class_true, exclude_ignored_pick), tf.gather_nd(class_pred, exclude_ignored_pick)
        bbox_true_valid, bbox_pred_valid = tf.gather_nd(bbox_true, valid_pick), tf.gather_nd(bbox_pred, valid_pick)

        cls_loss = self.__focal_loss__(class_true_valid, class_pred_valid)  # divide before sum, if meet inf
        bbox_loss = self.__bbox_loss__(bbox_true_valid, bbox_pred_valid)
        cls_loss, bbox_loss = tf.reduce_sum(cls_loss) / num_positive_anchors, tf.reduce_sum(bbox_loss) / num_positive_anchors

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
class ClassAccuracyWithBbox(tf.keras.metrics.Metric):
    def __init__(self, name="cls_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cls_acc = self.add_weight(name="cls_acc", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pick = tf.where(y_true[:, :, -1] == 1)
        cls_true_valid = tf.argmax(tf.gather_nd(y_true[:, :, 4:-1], pick), axis=-1)
        cls_pred_valid = tf.argmax(tf.gather_nd(y_pred[:, :, 4:], pick), axis=-1)
        cls_acc = tf.reduce_mean(tf.cast(cls_true_valid == cls_pred_valid, "float32"))
        # tf.assert_less(cls_acc, 1.1)
        self.cls_acc.assign_add(cls_acc)
        self.count.assign_add(1.0)

    def result(self):
        return self.cls_acc / self.count


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class AnchorFreeLoss(tf.keras.losses.Loss):
    """
    Basic test:
    >>> from keras_cv_attention_models.coco import losses, anchors_func
    >>> anchors = anchors_func.get_anchors([640, 640], pyramid_levels=[3, 5], aspect_ratios=[1], num_scales=1, anchor_scale=1, grid_zero_start=True)
    >>> aa = losses.AnchorFreeLoss(anchors, use_l1_loss=True)

    >>> from keras_cv_attention_models import yolox, test_images
    >>> from keras_cv_attention_models.coco import anchors_func, data
    >>> mm = yolox.YOLOXS()
    >>> img = test_images.dog_cat()
    >>> pred = mm(mm.preprocess_input(img))

    >>> bbs, lls, ccs = mm.decode_predictions(pred)[0]
    >>> bbox_labels_true = tf.concat([bbs, tf.one_hot(lls, 80), tf.ones([bbs.shape[0], 1])], axis=-1)
    >>> print(aa(tf.expand_dims(bbox_labels_true, 0), pred))
    >>> # tf.Tensor(2.3482049, shape=(), dtype=float32)
    """
    def __init__(
        self,
        anchors,
        use_l1_loss=False,
        bbox_loss_weight=5.0,
        anchor_assign_center_radius=2.5,
        anchor_assign_topk_ious_max=10,
        epsilon=1e-8,
        label_smoothing=0.0,
        from_logits=False,
        **kwargs
    ):
        from keras_cv_attention_models.coco import anchors_func

        super().__init__(**kwargs)
        self.bbox_loss_weight, self.use_l1_loss, self.epsilon = bbox_loss_weight, use_l1_loss, epsilon
        self.label_smoothing, self.from_logits = label_smoothing, from_logits
        self.anchors = anchors
        self.anchor_assign = anchors_func.AnchorFreeAssignMatching(anchors, anchor_assign_center_radius, anchor_assign_topk_ious_max, epsilon=epsilon)

    def __iou_loss__(self, bboxes_trues, bboxes_preds):
        # bboxes_trues: [[top, left, bottom, right]], bboxes_preds: [[top, left, bottom, right]]
        inter_top_left = tf.maximum(bboxes_trues[:, :2], bboxes_preds[:, :2])
        inter_bottom_right = tf.minimum(bboxes_trues[:, 2:], bboxes_preds[:, 2:])
        inter_hw = tf.maximum(inter_bottom_right - inter_top_left, 0)
        inter_area = inter_hw[:, 0] * inter_hw[:, 1]

        bboxes_trues_area = (bboxes_trues[:, 2] - bboxes_trues[:, 0]) * (bboxes_trues[:, 3] - bboxes_trues[:, 1])
        bboxes_preds_area = (bboxes_preds[:, 2] - bboxes_preds[:, 0]) * (bboxes_preds[:, 3] - bboxes_preds[:, 1])
        union_area = bboxes_trues_area + bboxes_preds_area - inter_area
        iou = inter_area / (union_area + self.epsilon)
        return 1 - iou ** 2

    def __valid_call_single__(self, bbox_labels_true, bbox_labels_pred):
        bboxes_true, bboxes_true_encoded, labels_true, object_true, bboxes_pred, bboxes_pred_encoded, labels_pred = self.anchor_assign(
            bbox_labels_true, bbox_labels_pred
        )

        num_valid_anchors = tf.cast(tf.shape(bboxes_pred)[0], bboxes_pred.dtype)
        if self.label_smoothing > 0:
            labels_true = labels_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        class_loss = tf.reduce_sum(K.binary_crossentropy(labels_true, labels_pred)) / num_valid_anchors
        object_loss = tf.reduce_sum(K.binary_crossentropy(object_true, bbox_labels_pred[:, -1])) / num_valid_anchors
        bbox_loss = tf.reduce_sum(self.__iou_loss__(bboxes_true, bboxes_pred)) / num_valid_anchors
        if self.use_l1_loss:
            l1_loss = tf.reduce_sum(tf.abs(bboxes_true_encoded - bboxes_pred_encoded)) / num_valid_anchors    # mean absolute error
        else:
            l1_loss = 0.0
        # return class_loss + object_loss + l1_loss + bbox_loss * self.bbox_loss_weight
        return class_loss, bbox_loss, object_loss, l1_loss

    def __call_single__(self, inputs):
        bbox_labels_true, bbox_labels_pred = inputs[0], inputs[1]
        return tf.cond(
            tf.reduce_any(bbox_labels_true[:, -1] > 0),
            lambda: self.__valid_call_single__(bbox_labels_true, bbox_labels_pred),
            lambda: (0.0, 0.0, tf.reduce_mean(K.binary_crossentropy(0.0, bbox_labels_pred[:, -1])), 0.0),    # Object loss only, object_trues is all False.
        )

    def call(self, y_true, y_pred):
        if self.from_logits:
            bbox_pred, class_pred = y_pred[:, :, :4], y_pred[:, :, 4:]
            class_pred = tf.sigmoid(class_pred)
            y_pred = tf.concat([bbox_pred, class_pred], axis=-1)

        class_loss, bbox_loss, object_loss, l1_loss = tf.map_fn(self.__call_single__, (y_true, y_pred), fn_output_signature=(y_pred.dtype,) * 4)

        class_loss, bbox_loss, object_loss = tf.reduce_mean(class_loss), tf.reduce_mean(bbox_loss), tf.reduce_mean(object_loss)
        l1_loss = tf.reduce_mean(l1_loss)
        tf.print(" - cls_loss:", class_loss, "- bbox_loss:", bbox_loss, "- obj_loss:", object_loss, end="\r")
        return class_loss + object_loss + l1_loss + bbox_loss * self.bbox_loss_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bbox_loss_weight": self.bbox_loss_weight,
                "use_l1_loss": self.use_l1_loss,
                "epsilon": self.epsilon,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
                "__anchors__": {
                    "input_shape": self.anchors.input_shape,
                    "pyramid_levels": self.anchors.pyramid_levels,
                    "aspect_ratios": self.anchors.aspect_ratios,
                    "num_scales": self.anchors.num_scales,
                    "anchor_scale": self.anchors.anchor_scale,
                    "grid_zero_start": self.anchors.grid_zero_start,
                }
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        from keras_cv_attention_models.coco import anchors_func
        config["anchors"] = anchors_func.get_anchors(**config.pop("__anchors__"))
        return cls(**config)
