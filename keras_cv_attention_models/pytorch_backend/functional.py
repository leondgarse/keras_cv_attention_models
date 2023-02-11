import torch
import math
import torch.nn.functional as F
from keras_cv_attention_models.pytorch_backend.layers import Lambda, Concatenate, GraphNode, Shape, _ZeroPadding, Add
from functools import partial


# eye, tf.image.extract_patches, shape, tile, stack


def wrapper(func, inputs, name=None):
    return Lambda(func, name=name)(inputs) if isinstance(inputs, GraphNode) or (isinstance(inputs, list) and isinstance(inputs[0], GraphNode)) else func(inputs)


def assign(parameter, data):
    parameter.data = torch.tensor(data, dtype=parameter.dtype)


def cast(inputs, dtype="float32"):
    return convert_to_tensor(inptus, dtype)


def clip_by_value(inputs, clip_value_min, clip_value_max, name=None):
    return wrapper(partial(torch.clip, min=clip_value_min, max=clip_value_max), inputs, name=name)


def concat(inputs, axis, name=None):
    return Concatenate(axis=axis, name=name)(inputs)


def convert_to_tensor(inputs, dtype="float32"):
    return torch.tensor(inputs, dtype=getattr(torch, dtype))


def cos(inputs, name=None):
    return wrapper(torch.cos, inputs, name=name)


def exp(inputs, name=None):
    return wrapper(torch.exp, inputs, name=name)


def expand_dims(inputs, axis, name=None):
    return wrapper(partial(torch.unsqueeze, dim=axis), inputs, name=name)


def extract_patches(inputs, sizes, strides=1, rates=1, padding=0, name=None):
    """
    >>> import torch
    >>> from torch import nn

    >>> image = np.zeros([256, 256, 3])
    >>> image[:, :, 1] += 128 # G
    >>> image[:, :, 2] += 256 # B

    >>> aa = np.expand_dims(image.astype("float32"), 0)
    >>> tf_out = tf.image.extract_patches(aa, sizes=[1, 3, 3, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='VALID')

    >>> kernel_size = 3
    >>> # cc = nn.Unfold(kernel_size=kernel_size, padding=0, stride=2)(torch.from_numpy(aa).permute(0, 3, 1, 2))
    >>> cc = torch.functional.F.unfold(torch.from_numpy(aa).permute(0, 3, 1, 2), kernel_size=kernel_size, padding=0, stride=2)
    >>> patch_shape = list(torch.functional.F.conv2d(torch.zeros([0, 3, 256, 256]), torch.zeros([3, 3, 3, 3]), stride=2).shape[2:])
    >>> cc = cc.reshape([-1, aa.shape[-1], kernel_size * kernel_size, cc.shape[-1]]).permute([0, 3, 2, 1])
    >>> torch_out = cc.reshape([-1, *patch_shape, kernel_size * kernel_size * aa.shape[-1]])
    >>> print(f"{np.allclose(tf_out, torch_out.detach()) = }")
    >>> # np.allclose(tf_out, torch_out.detach()) = True
    """
    unfold = F.unfold(inputs, kernel_size=sizes, dilation=rates, padding=padding, stride=strides)
    unfold = unfold.reshape([-1, inputs.shape[1], kernel_size * kernel_size, unfold.shape[-1]]).permute([0, 3, 2, 1])


def gather(inputs, indices, axis=None, batch_dims=0, name=None):
    """Defaults axis=None means the first non-batch dimension"""
    axis = batch_dims if axis is None else (len(inputs.shape) + axis if axis < 0 else axis)
    indices = tuple([slice(None)] * axis + [indices])
    return inputs[indices]  # .contiguous()


def gelu(inputs, approximate=False, name=None):
    return wrapper(partial(F.gelu, approximate="tanh" if approximate else "none"), inputs, name=name)


def l2_normalize(inputs, axis=None, epsilon=1e-12, name=None):
    return wrapper(partial(F.normalize, p=2.0, dim=axis, eps=epsilon), inputs, name=name)


def linspace(start, stop, num, name=None, axis=0):
    # return Lambda(partial(torch.linspace, start=start, end=stop, steps=num), name=name)(inputs)
    return torch.linspace(start=start, end=stop, steps=num)


def log(log, name=None):
    return wrapper(partial(torch.log), inputs, name=name)


def matmul(xx, yy, name=None):
    return wrapper(lambda inputs: torch.matmul(inputs[0], inputs[1]), [xx, yy], name=name)


def moments(inputs, axes, shift=None, keepdims=False, name=None):
    return wrapper(partial(torch.var_mean, dim=axes, keepdim=keepdims), inputs, name=name)


def maximum(xx, yy, name=None):
    if isinstance(yy, torch.Tensor):
        func = lambda inputs: torch.maximum(inputs[0], inputs[1])
    else:  # maximum doesn't support scalar value
        func = lambda inputs: torch.where(inputs[0] > inputs[1], inputs[0], inputs[1])
    return wrapper(func, [xx, yy], name=name)


def minimum(xx, yy, name=None):
    if isinstance(yy, torch.Tensor):
        func = lambda inputs: torch.minimum(inputs[0], inputs[1])
    else:  # maximum doesn't support scalar value
        func = lambda inputs: torch.where(inputs[0] > inputs[1], inputs[1], inputs[0])
    return wrapper(func, [xx, yy], name=name)


def norm(inputs, ord="euclidean", axis=1, keepdims=False, name=None):
    return wrapper(partial(torch.norm, p=2, dim=axis, keepdim=keepdims), inputs, name=name)


def pad(inputs, paddings, mode="CONSTANT", constant_values=0, name=None):
    """
    torch pad is like `[left, right, top, bottom]`
    >>> aa = tf.pad(np.array([[[1, 2, 3], [4, 5, 6]]]), [[0, 0], [1, 2], [3, 4]])
    >>> bb = torch.functional.F.pad(torch.tensor([[[1, 2, 3], [4, 5, 6]]]), [3, 4, 1, 2, 0, 0])
    >>> np.allclose(aa, bb.detach())
    """
    # F.pad doesn't support 0 shape inputs, throws error while compute_output_shape
    # pad = []
    # for pp in paddings[::-1]:
    #     pad += pp
    # return Lambda(partial(F.pad, pad=pad, mode=mode.lower(), value=constant_values), name=name)(inputs)
    return _ZeroPadding(padding=paddings)(inputs)


def pow(inputs, exponent, name=None):
    return wrapper(partial(torch.pow, exponent=exponent), inputs, name=name)


def reduce_max(inputs, axis=None, keepdims=False, name=None):
    return wrapper(partial(torch.max, dim=axis, keepdim=keepdims), inputs, name=name)


def reduce_mean(inputs, axis=None, keepdims=False, name=None):
    return wrapper(partial(torch.mean, dim=axis, keepdim=keepdims), inputs, name=name)


def reduce_sum(inputs, axis=None, keepdims=False, name=None):
    if isinstance(inputs, (list, tuple)) and axis == 0:
        return Add(name=name)(inputs)
    else:
        return wrapper(partial(torch.sum, dim=axis, keepdim=keepdims), inputs, name=name)


def relu(inputs, name=None):
    return wrapper(F.relu, inputs, name=name)


def relu6(inputs, name=None):
    return wrapper(F.relu6, inputs, name=name)


def repeat(inputs, repeats, axis, name=None):
    """
    >>> aa = np.arange(6).reshape(2, 3)
    >>> torch_out = torch.expand_copy(torch.unsqueeze(torch.from_numpy(aa), 1), [2, 2, 3]).reshape(4, 3)
    >>> tf_out = tf.repeat(aa, repeats=2, axis=0)
    >>> np.allclose(torch_out.detach(), tf_out)
    """
    if inputs.shape[axis] is None:
        expand_shape = [-1] * len(inputs.shape)
        expand_shape.insert(axis + 1, repeats)
        out_shape = [(-1 if ii is None or ii == -1 else ii * repeats) if dim == axis else ii for dim, ii in enumerate(inputs.shape)]
        return wrapper(lambda inputs: torch.expand_copy(torch.unsqueeze(inputs, axis + 1), expand_shape).contiguous().view(out_shape), inputs, name=name)
    else:
        return wrapper(partial(torch.repeat_interleave, repeats=repeats, dim=axis), inputs, name=name)
    # expand_shape.insert(axis + 1, repeats)
    # expand_shape = tuple(expand_shape)
    # out_shape = [(-1 if ii is None or ii == -1 else ii * repeats) if dim == axis else ii for dim, ii in enumerate(inputs.shape)]
    # return wrapper(lambda inputs: torch.reshape(torch.expand_copy(torch.unsqueeze(inputs, axis + 1), expand_shape), out_shape), inputs, name=name)


def reshape(inputs, shape, name=None):
    # return wrapper(partial(torch.reshape, shape=shape), inputs, name=name)
    return wrapper(lambda inputs: inputs.contiguous().view(shape), inputs, name=name)


def resize(inputs, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    if isinstance(inputs, GraphNode):
        return Lambda(partial(F.interpolate, size=size, mode=method, antialias=antialias), name=name)(inputs)  # [TODO] align_corners
    else:  # called directly
        inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)
        if len(inputs.shape) == 3:
            return F.interpolate(inputs[None], size=size, mode=method, antialias=antialias)[0]
        else:
            return F.interpolate(inputs, size=size, mode=method, antialias=antialias)


def shape(inputs):
    # return Shape()(inputs)
    return inputs.shape


def sigmoid(inputs, axis=None, name=None):
    return wrapper(torch.sigmoid, inputs, name=name)


def sin(inputs, name=None):
    return wrapper(torch.sin, inputs, name=name)


def softmax(inputs, axis=None, name=None):
    return wrapper(partial(torch.softmax, dim=axis), inputs, name=name)


def softplus(inputs, name=None):
    return wrapper(F.softplus, inputs, name=name)


def split(inputs, num_or_size_splits, axis=0, num=None, name=None):
    axis = len(inputs.shape) - 1 if axis == -1 else axis
    split_axis_shape = inputs.shape[axis]
    assert split_axis_shape is not None

    if isinstance(num_or_size_splits, int):
        # split_axis_shape, num_or_size_splits = 5, 3 -> size_splits [2, 2, 1]
        split_size = math.ceil(split_axis_shape / num_or_size_splits)
        size_splits = [split_size] * num_or_size_splits  # doesn't matter if the last one exceeding split_axis_shape
    else:
        size_splits = num_or_size_splits
        size_splits = [0 if ii is None or ii == -1 else ii for ii in size_splits]
        num_unknown_dim = sum([ii == 0 for ii in size_splits])
        assert num_unknown_dim < 2, "At most one unknown dimension in num_or_size_splits: {}".format(num_or_size_splits)

        if num_unknown_dim == 1:
            size_splits = [(split_axis_shape - sum(size_splits)) if ii == 0 else ii for ii in size_splits]
    # [112, 112] -> [slice(0, 112), slice(112, 224)]
    split_slices = [slice(int(sum(size_splits[:id])), sum(size_splits[: id + 1])) for id, ii in enumerate(size_splits)]

    pre_axis_slice = [slice(None)] * axis
    return [inputs[tuple([*pre_axis_slice, split_slice])] for split_slice in split_slices]


def sqrt(inputs, name=None):
    return wrapper(torch.sqrt, inputs, name=name)


def squeeze(inputs, axis, name=None):
    return wrapper(partial(torch.squeeze, dim=axis), inputs, name=name)


def tanh(inputs, name=None):
    return wrapper(F.tanh, inputs, name=name)


def transpose(inputs, perm=None, conjugate=False, name=None):
    return wrapper(partial(torch.permute, dims=perm), inputs, name=name)


def unstack(inputs, axis, name=None):
    axis = len(inputs.shape) + axis if axis < 0 else axis
    axis_shape = inputs.shape[axis]
    assert axis_shape is not None

    pre_axis_slice = [slice(None)] * axis
    return [inputs[tuple([*pre_axis_slice, index])] for index in range(axis_shape)]
