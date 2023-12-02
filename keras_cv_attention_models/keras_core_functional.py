import keras_core as keras
from keras_core.ops import *
from keras_core.ops import concatenate as concat
from keras_core.ops import mean as reduce_mean
from keras_core.ops import max as reduce_max
from keras_core.ops import min as reduce_min
from keras_core.ops import power as pow
from keras_core.ops import clip as clip_by_value


def extract_patches(images, sizes=1, strides=1, rates=1, padding="valid", name=None):
    return keras.ops.image.extract_patches(
        images,
        size=sizes[1:-1] if isinstance(sizes, int) or len(sizes) > 2 else sizes,
        strides=strides[1:-1] if isinstance(strides, int) or len(strides) > 2 else strides,
        dilation_rate=rates[1:-1] if isinstance(rates, int) or len(rates) > 2 else rates,
        padding=padding.lower(),
        data_format=keras.backend.image_data_format(),
    )


def gather(inputs, indices, axis=None, batch_dims=0, name=None):
    """Defaults axis=None means the first non-batch dimension"""
    axis = batch_dims if axis is None else (len(inputs.shape) + axis if axis < 0 else axis)
    return keras.ops.take(inputs, indices, axis=axis)


def l2_normalize(inputs, axis=None, epsilon=1e-12, name=None):
    return inputs / keras.ops.sqrt(keras.ops.maximum(keras.ops.sum(inputs**2, axis=axis, keepdims=True), epsilon))


def norm(inputs, ord="euclidean", axis=1, keepdims=False, name=None):
    return keras.ops.sqrt(keras.ops.sum(inputs**2, axis=axis, keepdims=True))


def resize(images, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    return keras.ops.image.resize(images, size, interpolation=method, antialias=antialias, data_format=keras.backend.image_data_format())


def reduce_sum(inputs, axis=None, keepdims=False, name=None):
    axis = () if axis is None else axis
    if isinstance(inputs, (list, tuple)) and axis == 0:
        rr = inputs[0]
        for ii in inputs[1:]:
            rr += ii
        return rr
    else:
        # return wrapper(lambda xx: xx.sum(dim=axis, keepdim=keepdims), inputs, name=name)
        return keras.ops.sum(inputs, axis=axis, keepdims=keepdims)


def rsqrt(inputs, name=None):
    return keras.ops.true_divide(1, keras.ops.sqrt(inputs))


def split(inputs, num_or_size_splits, axis=0, num=None, name="split"):
    from builtins import sum

    if isinstance(num_or_size_splits, int):
        return keras.ops.split(inputs, num_or_size_splits, axis=axis)

    axis = (len(inputs.shape) + axis) if axis < 0 else axis
    split_axis_shape = inputs.shape[axis]
    assert split_axis_shape is not None

    size_splits = num_or_size_splits
    size_splits = [0 if ii is None or ii == -1 else ii for ii in size_splits]
    num_unknown_dim = sum([ii == 0 for ii in size_splits])
    assert num_unknown_dim < 2, "At most one unknown dimension in num_or_size_splits: {}".format(num_or_size_splits)

    if num_unknown_dim == 1:
        size_splits = [(split_axis_shape - sum(size_splits)) if ii == 0 else ii for ii in size_splits]

    cum_split = [sum(size_splits[: id + 1]) for id, _ in enumerate(size_splits)]
    # len(keras.ops.split(np.ones([2, 6]), [2, 2, 2], axis=-1)) == 4
    # len(keras.ops.split(keras.layers.Input([6]), [2, 2, 2], axis=-1)) == 3
    return keras.ops.split(inputs, cum_split, axis=axis)[: len(size_splits)]
