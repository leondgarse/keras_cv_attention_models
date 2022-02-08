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
