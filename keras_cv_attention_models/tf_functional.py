from tensorflow.nn import *
from tensorflow.math import *

from tensorflow import (
    cast,
    clip_by_value,
    concat,
    convert_to_tensor,
    expand_dims,
    eye,
    gather,
    linspace,
    matmul,
    norm,
    pad,
    repeat,
    reshape,
    shape,
    split,
    squeeze,
    stack,
    tile,
    transpose,
    unstack,
)
from tensorflow.image import resize, extract_patches


def assign(parameter, data):
    parameter.assign(data)
