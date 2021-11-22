import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class BinaryCrossEntropyTimm(tf.keras.losses.BinaryCrossentropy):
    """
    >>> import torch, timm.loss
    >>> from keras_cv_attention_models.imagenet import losses
    >>> tt = timm.loss.BinaryCrossEntropy(smoothing=0.0, target_threshold=0.2)
    >>> ss = losses.BinaryCrossEntropyTimm(target_threshold=0.2, from_logits=True)
    >>> y_true = tf.one_hot(np.random.permutation(20).reshape(2, 10), 10).numpy()
    >>> y_true = np.clip(y_true[0] + y_true[1], 0, 1)
    >>> y_pred = np.random.uniform(size=(10, 10))
    >>> torch_out = tt(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy()
    >>> keras_out = ss(y_true, y_pred).numpy()
    >>> print(f"{torch_out = }, {keras_out = }")
    # torch_out = array(0.9457581, dtype=float32), keras_out = 0.945758044719696
    """

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
