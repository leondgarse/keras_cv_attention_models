import keras_core
from keras_core.ops import *
from keras_core.ops import concatenate as concat
from keras_core.ops import mean as reduce_mean
from keras_core.ops import sum as reduce_sum
from keras_core.ops import max as reduce_max
from keras_core.ops import min as reduce_min
from keras_core.ops import power as pow
from keras_core.ops import clip as clip_by_value
from keras_core.ops.image import extract_patches


def resize(images, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    return keras_core.ops.image.resize(images, size, interpolation=method, antialias=antialias, data_format=keras_core.backend.image_data_format())


def split(inputs, num_or_size_splits, axis=0, num=None, name="split"):
    if isinstance(num_or_size_splits, int):
        return keras_core.ops.split(inputs, num_or_size_splits, axis=axis)

    axis = (len(inputs.shape) + axis) if axis < 0 else axis
    split_axis_shape = inputs.shape[axis]
    assert split_axis_shape is not None

    size_splits = num_or_size_splits
    size_splits = [0 if ii is None or ii == -1 else ii for ii in size_splits]
    num_unknown_dim = sum([ii == 0 for ii in size_splits])
    assert num_unknown_dim < 2, "At most one unknown dimension in num_or_size_splits: {}".format(num_or_size_splits)

    if num_unknown_dim == 1:
        size_splits = [(split_axis_shape - sum(size_splits)) if ii == 0 else ii for ii in size_splits]

    cum_split = [sum(num_or_size_splits[: id + 1]) for id, _ in enumerate(size_splits[:-1])]
    return keras_core.ops.split(inputs, cum_split, axis=axis)
