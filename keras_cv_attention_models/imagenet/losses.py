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
        super().__init__(label_smoothing=label_smoothing, **kwargs)
        self.target_threshold = target_threshold
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        target_threshold = tf.cast(self.target_threshold, y_true.dtype)
        y_true = tf.where(y_true > target_threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
        return super().call(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update({"target_threshold": self.target_threshold, "label_smoothing": self.label_smoothing})
        return config


@tf.keras.utils.register_keras_serializable(package="kecamLoss")
class DistillKLDivergenceLoss(tf.keras.losses.Loss):
    """[PDF 2106.05237 Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/pdf/2106.05237.pdf)
    Modified according [Knowledge distillation recipes](https://keras.io/examples/keras_recipes/better_knowledge_distillation/)

    Temperature affecting:
    >>> teacher_prob = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    >>> _ = [print("temperature:", temp, tf.nn.softmax(teacher_prob / temp).numpy()) for temp in [0.1, 1, 10, 20]]
    >>> # temperature: 0.1 [3.92559586e-05 2.90064480e-04 2.14330272e-03 1.58369840e-02 1.17020363e-01 8.64670029e-01]
    >>> # temperature: 1 [0.09542741 0.11655531 0.14236097 0.17388009 0.21237762 0.25939861]
    >>> # temperature: 10 [0.1584458  0.16164661 0.16491209 0.16824354 0.17164228 0.17510968]
    >>> # temperature: 20 [0.16252795 0.16416138 0.16581123 0.16747766 0.16916084 0.17086094]
    """

    def __init__(self, temperature=10, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        # self.kl_divergence = tf.keras.losses.KLDivergence()

    def call(self, teacher_prob, student_prob):
        return tf.losses.kl_divergence(
            tf.nn.softmax(teacher_prob / self.temperature, axis=-1),
            tf.nn.softmax(student_prob / self.temperature, axis=-1),
        )


# Not using, from VOLO with mix_token lambda
def token_label_class_loss(y_true, y_pred):
    # tf.print(", y_true:", y_true.shape, "y_pred:", y_pred.shape, end="")
    if y_pred.shape[-1] != y_true.shape[-1]:
        y_pred, cls_lambda = y_pred[:, :-1], y_pred[:, -1:]
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = cls_lambda * y_true + (1 - cls_lambda) * y_true[::-1]
    return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
