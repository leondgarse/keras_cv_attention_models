import os

is_tensorflow_backend = not "torch" in os.getenv("KECAM_BACKEND", "tensorflow").lower()

if is_tensorflow_backend:
    import tensorflow as tf
    from tensorflow.keras import layers, models, initializers
    from tensorflow.keras.utils import register_keras_serializable
    from keras_cv_attention_models import tf_functional as functional
else:
    from keras_cv_attention_models.pytorch_backend import layers, models, functional, initializers
    from keras_cv_attention_models.pytorch_backend.utils import register_keras_serializable


def backend():
    if is_tensorflow_backend:
        return tf.keras.backend.backend()
    else:
        return "pytorch"


def image_data_format():
    if is_tensorflow_backend:
        return tf.keras.backend.image_data_format()
    else:
        return "channels_first"


def in_train_phase(train_phase, eval_phase, training=None):
    if is_tensorflow_backend:
        return tf.keras.backend.in_train_phase(train_phase, eval_phase, training=training)
    else:
        return train_phase if training else eval_phase  # [TODO]


def numpy_image_resize(inputs, target_shape, method="nearest", is_source_channels_last=True):
    ndims = len(inputs.shape)
    if ndims < 2 or ndims > 4:
        raise ValueError("inputs with shape={}, ndims={} not supported, ndims must in [2, 4]".format(ipnuts.shape, ndims))

    if is_source_channels_last:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, :, :, None])
        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 3, 1, 2])

        inputs = functional.resize(inputs, target_shape, method=method)
        inputs = inputs.detach().numpy() if hasattr(inputs, "detach") else inputs.numpy()

        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 2, 3, 1])
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, :, :, 0])
    else:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, None, :, :])
        inputs = inputs.transpose([0, 2, 3, 1]) if image_data_format() == "channels_last" else inputs

        inputs = functional.resize(inputs, target_shape, method=method)
        inputs = inputs.detach().numpy() if hasattr(inputs, "detach") else inputs.numpy()

        inputs = inputs.transpose([0, 3, 1, 2]) if image_data_format() == "channels_last" else inputs
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, 0])
    return inputs
