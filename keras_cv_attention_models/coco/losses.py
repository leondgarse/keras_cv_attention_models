import tensorflow as tf
from tensorflow.keras import backend as K

# @tf.keras.utils.register_keras_serializable(package="kecamLoss")
class FocalLossWithBbox(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, sigma=3.0, regression_loss_weight=50, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.gamma, self.sigma, self.regression_loss_weight = alpha, gamma, sigma, regression_loss_weight
        self.squared_sigma = sigma ** 2

    def __classification_loss__(self, class_true, class_pred, anchor_mark):
        exclude_ignored_pick = tf.where(anchor_mark != -1)
        num_positive_anchors = tf.maximum(tf.reduce_sum(tf.cast(anchor_mark == 1, class_pred.dtype)), 1)

        class_true_valid = tf.gather_nd(class_true, exclude_ignored_pick)
        class_pred_valid = tf.gather_nd(class_pred, exclude_ignored_pick)
        # focal_factor = (1 - output) ** gamma if class is 1 else output ** gamma
        focal_bce = K.binary_focal_crossentropy(class_true_valid, class_pred_valid, gamma=self.gamma, from_logits=True)
        # 1 -> 0.25, 0 -> 0.75
        alpha_factor = class_true_valid * self.alpha + (1 - class_true_valid) * (1 - self.alpha)
        classification_loss = tf.reduce_sum(alpha_factor * focal_bce) / num_positive_anchors
        return classification_loss

    def __regression_loss__(self, bbox_true, bbox_pred, anchor_mark):
        valid_pick = tf.where(anchor_mark == 1)
        bbox_true_valid = tf.gather_nd(bbox_true, valid_pick)
        bbox_pred_valid = tf.gather_nd(bbox_pred, valid_pick)
        l1_dists = tf.abs(bbox_true_valid - bbox_pred_valid)
        regression_loss = tf.where(l1_dists < 1 / self.squared_sigma, self.squared_sigma * 0.5 * (l1_dists ** 2), l1_dists - 0.5 / self.squared_sigma)
        regression_loss = tf.reduce_mean(regression_loss)
        return regression_loss

    def call(self, y_true, y_pred):
        bbox_pred, class_pred = y_pred[:, :, :4], y_pred[:, :, 4:]
        bbox_true, class_true, anchor_mark = y_true[:, :, :4], y_true[:, :, 4:-1], y_true[:, :, -1]

        cls_loss = self.__classification_loss__(class_true, class_pred, anchor_mark)
        bbox_loss = self.__regression_loss__(bbox_true, bbox_pred, anchor_mark)
        return cls_loss + bbox_loss * self.regression_loss_weight

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma, "sigma": self.sigma, "regression_loss_weight": self.regression_loss_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
