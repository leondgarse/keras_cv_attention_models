import os

is_tensorflow_backend = not "torch" in os.getenv("KECAM_BACKEND", "tensorflow").lower()

if is_tensorflow_backend:
    import tensorflow as tf
    from tensorflow.keras import layers, models, initializers
    from tensorflow.keras.utils import register_keras_serializable
    import tensorflow.nn as functional

    def __set_functional_attr__(func, name=None):
        setattr(functional, name if name else func.__name__, func)

    __set_functional_attr__(tf.clip_by_value)
    __set_functional_attr__(tf.reduce_mean)
    __set_functional_attr__(tf.reduce_sum)
    __set_functional_attr__(tf.reshape)
    __set_functional_attr__(tf.unstack)
    __set_functional_attr__(tf.split)
    __set_functional_attr__(tf.cos)
    __set_functional_attr__(tf.sin)
    __set_functional_attr__(tf.norm, "norm")
    __set_functional_attr__(tf.squeeze, "squeeze")
    __set_functional_attr__(tf.image.resize, "resize")
    __set_functional_attr__(tf.expand_dims, "expand_dims")
    __set_functional_attr__(tf.convert_to_tensor, "convert_to_tensor")
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


def in_train_phase():
    if is_tensorflow_backend:
        return tf.keras.backend.in_train_phase()
    else:
        return True  # [TODO]
