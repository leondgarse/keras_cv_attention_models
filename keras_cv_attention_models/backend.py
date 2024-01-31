import os

is_tensorflow_backend, is_torch_backend, is_jax_backend, is_keras_core_backend = False, False, False, False
if "tensorflow" in os.getenv("KECAM_BACKEND", "tensorflow").lower():
    is_tensorflow_backend = True
elif "torch" in os.getenv("KECAM_BACKEND", "tensorflow").lower():
    is_torch_backend = True
elif "jax" in os.getenv("KECAM_BACKEND", "tensorflow").lower():
    is_jax_backend = True
    is_keras_core_backend = True
    os.environ["KERAS_BACKEND"] = "jax"
else:
    is_keras_core_backend = True

if is_tensorflow_backend:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as ee:
        print(">>>> [Warning] os environ 'KECAM_BACKEND' is 'tensorflow', but not installed.")
        try:
            import torch

            is_tensorflow_backend = False
            is_torch_backend = True
        except ModuleNotFoundError as ee:
            raise ModuleNotFoundError("Neither tensorflow nor torch found")


if is_tensorflow_backend:
    import tensorflow as tf
    from tensorflow.keras import layers, models, initializers, callbacks, metrics, losses, optimizers
    from tensorflow.keras.utils import register_keras_serializable, get_file
    from keras_cv_attention_models import tf_functional as functional
elif is_torch_backend:
    from keras_cv_attention_models.pytorch_backend import layers, models, functional, initializers, callbacks, metrics, losses, optimizers
    from keras_cv_attention_models.pytorch_backend.utils import register_keras_serializable, get_file

    print(">>>> Using PyTorch backend")
else:
    import keras_core
    from keras_core import layers, models, initializers, callbacks, metrics
    from keras_core.utils import register_keras_serializable, get_file

    # from keras_core import ops as functional
    from keras_cv_attention_models import keras_core_functional as functional


def backend():
    """backend() value can still be "torch" for `keras_core` backend, even `is_torch_backend` is False"""
    if is_tensorflow_backend:
        return tf.keras.backend.backend()
    elif is_torch_backend:
        return "torch"
    else:
        return keras_core.backend.backend()


def image_data_format():
    if is_tensorflow_backend:
        return tf.keras.backend.image_data_format()
    elif is_torch_backend:
        return "channels_first"
    else:
        return keras_core.backend.image_data_format()


__is_channels_last__ = image_data_format() == "channels_last"


def is_channels_last():
    return __is_channels_last__


def align_input_shape_by_image_data_format(input_shape):
    """Regard input_shape as force using original shape if len(input_shape) == 4,
    else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format,
    or if dynamic shape like `(None, None, 3)`, will regard the known shape 3 as channel dimension.

    Examples:
    >>> os.environ['KECAM_BACKEND'] = "torch"
    >>> from keras_cv_attention_models import backend
    >>> print(f"{backend.image_data_format() = }")
    >>> # backend.image_data_format() = 'channels_first'
    >>> print(backend.align_input_shape_by_image_data_format([224, 224, 3]))
    >>> # [3, 224, 224]
    >>> print(backend.align_input_shape_by_image_data_format([3, 224, 224]))
    >>> # [3, 224, 224]
    >>> print(backend.align_input_shape_by_image_data_format([None, None, 3]))
    >>> # [3, None, None]
    >>> print(backend.align_input_shape_by_image_data_format([None, 224, 224, 3]))
    >>> # [224, 224, 3]
    """
    # channel_axis = if image_data_format() == "channels_last" else
    if len(input_shape) == 4:  # Regard this as force using original shape
        return input_shape[1:]

    # Assume channel dimension is the one with min value in input_shape
    if None in input_shape:
        orign_channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1] is None)
    else:
        orign_channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1])
    orign_block_shape = [dim for axis, dim in enumerate(input_shape) if axis != orign_channel_axis]
    aligned = [*orign_block_shape, channel_dim] if image_data_format() == "channels_last" else [channel_dim, *orign_block_shape]
    if aligned[1] != input_shape[1] or aligned[-1] != input_shape[-1]:
        print(">>>> Aligned input_shape:", aligned)
    return aligned


def in_train_phase(train_phase, eval_phase, training=None):
    if is_tensorflow_backend:
        return tf.keras.backend.in_train_phase(train_phase, eval_phase, training=training)
    elif is_torch_backend:
        return train_phase if training else eval_phase  # [TODO]
    else:
        return train_phase if training else eval_phase  # [TODO]


def numpy_image_resize(inputs, target_shape, method="bilinear", antialias=False, is_source_channels_last=True):
    import numpy as np

    ndims = len(inputs.shape)
    if ndims < 2 or ndims > 4:
        raise ValueError("inputs with shape={}, ndims={} not supported, ndims must in [2, 4]".format(ipnuts.shape, ndims))

    target_shape = [int(ii) for ii in target_shape]
    if is_source_channels_last:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, :, :, None])
        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 3, 1, 2])

        inputs = functional.resize(inputs, target_shape, method=method, antialias=antialias)
        inputs = inputs.detach() if hasattr(inputs, "detach") else inputs
        inputs = inputs.numpy() if hasattr(inputs, "numpy") else np.array(inputs)

        inputs = inputs if image_data_format() == "channels_last" else inputs.transpose([0, 2, 3, 1])
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, :, :, 0])
    else:
        inputs = inputs if ndims == 4 else (inputs[None] if ndims == 3 else inputs[None, None, :, :])
        inputs = inputs.transpose([0, 2, 3, 1]) if image_data_format() == "channels_last" else inputs

        inputs = functional.resize(inputs, target_shape, method=method, antialias=antialias)
        inputs = inputs.detach() if hasattr(inputs, "detach") else inputs
        inputs = inputs.numpy() if hasattr(inputs, "numpy") else np.array(inputs)

        inputs = inputs.transpose([0, 3, 1, 2]) if image_data_format() == "channels_last" else inputs
        inputs = inputs if ndims == 4 else (inputs[0] if ndims == 3 else inputs[0, 0])
    return inputs
