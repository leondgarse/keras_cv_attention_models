import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class BinaryCrossEntropyTimm(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, target_threshold=0.0, label_smoothing=0.0, **kwargs):
        super(BinaryCrossEntropyTimm, self).__init__(label_smoothing=label_smoothing, **kwargs)
        self.target_threshold = target_threshold
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        target_threshold = tf.cast(self.target_threshold, y_true.dtype)
        y_true = tf.where(y_true > target_threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
        return super(BinaryCrossEntropyTimm, self).call(y_true, y_pred)

    def get_config(self):
        config = super(BinaryCrossEntropyTimm, self).get_config()
        config.update({"target_threshold": self.target_threshold, "label_smoothing": self.label_smoothing})
        return config
