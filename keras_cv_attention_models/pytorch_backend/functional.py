import torch
import math
import torch.nn.functional as F
from functools import partial
from keras_cv_attention_models.pytorch_backend.layers import (
    _ZeroPadding,
    _ReshapeDynamic,
    _ResizeDynamic,
    _GatherND,
    Add,
    Concatenate,
    Equal,
    GraphNode,
    Lambda,
    Shape,
    compute_conv_output_size,
)


def wrapper(func, inputs, output_shape=None, name=None):
    if isinstance(inputs, GraphNode) or (isinstance(inputs, list) and isinstance(inputs[0], GraphNode)):
        return Lambda(func, output_shape=output_shape, name=name)(inputs)
    else:
        return func(inputs)


def abs(inputs, name=None):
    return wrapper(torch.abs, inputs, name=name)


def argmax(inputs, axis=None, output_type="int64", name=None):
    axis = 0 if axis is None else (len(inputs.shape) + axis if axis < 0 else axis)
    output_shape = [ii for id, ii in enumerate(inputs.shape) if id != axis]
    return wrapper(partial(torch.argmax, dim=axis), inputs, output_shape=output_shape, name=name)


def argsort(inputs, axis=-1, direction="ASCENDING", stable=False, name=None):
    descending = direction.upper() == "DESCENDING"
    return wrapper(partial(torch.argsort, dim=axis, descending=descending, stable=stable), inputs, output_shape=inputs.shape, name=name)


def assign(parameter, data):
    parameter.data = torch.tensor(data, dtype=parameter.dtype)


def cast(inputs, dtype="float32"):
    if isinstance(inputs, torch.Tensor) and (inputs.dtype == dtype or str(inputs.dtype).endswith(str(dtype))):
        return inputs
    elif dtype == torch.float32 or dtype == "float32":
        return inputs.float()
    elif dtype == torch.float16 or dtype == "float16" or dtype == "half":
        return inputs.half()
    else:
        dtype = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
        return torch.tensor(inputs, dtype=getattr(torch, dtype))


def clip_by_value(inputs, clip_value_min, clip_value_max, name=None):
    if isinstance(clip_value_min, torch.Tensor) and not isinstance(clip_value_max, torch.Tensor):
        clip_value_max = torch.as_tensor(clip_value_max)  # Should be both Tensor or number
    elif isinstance(clip_value_max, torch.Tensor) and not isinstance(clip_value_min, torch.Tensor):
        clip_value_min = torch.as_tensor(clip_value_min)  # Should be both Tensor or number
    return wrapper(partial(torch.clip, min=clip_value_min, max=clip_value_max), inputs, name=name)


def concat(inputs, axis, name=None):
    return Concatenate(axis=axis, name=name)(inputs)


def convert_to_tensor(inputs, dtype="float32"):
    return torch.tensor(inputs, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype)


def cos(inputs, name=None):
    return wrapper(torch.cos, inputs, name=name)


def embedding_lookup(params, ids, max_norm=None, name=None):
    """
    >>> import math, torch
    >>> inputs = torch.randint(0, 1000, size=[32])
    >>> params = torch.rand([1000, 320])
    >>> print(np.allclose(torch.functional.F.embedding(inputs, params), tf.nn.embedding_lookup(params.numpy(), inputs.numpy())))
    # True
    """
    # return wrapper(partial(F.embedding, max_norm=max_norm)), [params, ids], name=name)
    return F.embedding(ids, params, max_norm=max_norm)


def equal(x, y, name=None):
    return Equal(name=name)([x, y]) if isinstance(y, GraphNode) else wrapper(lambda inputs: torch.eq(inputs, y), x, output_shape=x.shape, name=name)


def exp(inputs, name=None):
    return wrapper(torch.exp, inputs, name=name)


def expand_dims(inputs, axis, name=None):
    return wrapper(partial(torch.unsqueeze, dim=axis), inputs, name=name)


def extract_patches(inputs, sizes=1, strides=1, rates=1, padding="valid", data_format="channels_last", compressed=True, name=None):
    """
    tf.image.extract_patches -> out shape [batch, patch_height, patch_width, kernel_size * kernel_szie * channel]
    torch.functional.F.unfold -> out shape [batch, channel * kernel_size * kernel_szie, patch_height * patch_width]
    - compressed=False -> out shape [batch, patch_height, patch_width, kernel_size, kernel_szie, channel]

    >>> import torch
    >>> from keras_cv_attention_models.pytorch_backend import functional
    >>> kernel_size = 3
    >>> image = np.zeros([256, 256, 3])
    >>> image[:, :, 1] += 128 # G
    >>> image[:, :, 2] += 256 # B
    >>> aa = np.expand_dims(image.astype("float32"), 0)
    >>> tf_out = tf.image.extract_patches(aa, sizes=[1, kernel_size, kernel_size, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='valid')
    >>> torch_out = functional.extract_patches(torch.from_numpy(image[None].astype('float32')), kernel_size, 2)
    >>> print(f"{np.allclose(tf_out, torch_out.detach()) = }")
    """
    channel = inputs.shape[1] if data_format == "channels_first" else inputs.shape[-1]
    height, width = inputs.shape[2:] if data_format == "channels_first" else inputs.shape[1:-1]

    kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
    strides = strides[1] if isinstance(strides, (list, tuple)) else strides
    # dilation_rate can be 2 different values, used in DiNAT
    dilation_rate = (rates if len(rates) == 2 else rates[1:3]) if isinstance(rates, (list, tuple)) else (rates, rates)

    padding = padding.lower()
    pad_value = kernel_size // 2 if padding == "same" else 0
    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation_rate, padding=pad_value, stride=strides)
    patch_size = compute_conv_output_size((height, width), kernel_size, strides, padding, dilation_rate=dilation_rate)
    if data_format == "channels_first":
        if compressed:
            target_shape = [-1, channel * kernel_size * kernel_size, *patch_size]
        else:
            target_shape = [-1, channel, kernel_size, kernel_size, *patch_size]
        module = lambda xx: unfold(xx).contiguous().view(target_shape)
    else:
        pre_perm_shape = [-1, channel, kernel_size * kernel_size, patch_size[0] * patch_size[1]]
        perm = [0, 3, 2, 1]
        if compressed:
            target_shape = [-1, *patch_size, kernel_size * kernel_size * channel]
        else:
            target_shape = [-1, *patch_size, kernel_size, kernel_size, channel]
        module = lambda xx: unfold(xx.permute([0, 3, 1, 2])).contiguous().view(pre_perm_shape).permute(perm).contiguous().view(target_shape)
    return wrapper(module, inputs, name=name)


def gather(inputs, indices, axis=None, batch_dims=0, name=None):
    """Defaults axis=None means the first non-batch dimension"""
    axis = batch_dims if axis is None else (len(inputs.shape) + axis if axis < 0 else axis)
    indices = tuple([slice(None)] * axis + [indices])
    return inputs[indices]  # .contiguous()


def gather_nd(inputs, indices, batch_dims=0, name=None):
    # """Defaults axis=None means the first non-batch dimension"""
    # return inputs[indices.tolist()]  # .contiguous()
    # print(indices.shape[0])
    if isinstance(inputs, GraphNode) or isinstance(indices, GraphNode):
        return _GatherND(batch_dims=batch_dims, name=name)([inputs, indices])
    else:
        return inputs[indices.T.tolist()] if len(indices.shape) > 1 else inputs[tuple(indices.tolist())]  # .contiguous()


def gelu(inputs, approximate=False, name=None):
    return wrapper(partial(F.gelu, approximate="tanh" if approximate else "none"), inputs, name=name)


def l2_normalize(inputs, axis=None, epsilon=1e-12, name=None):
    return wrapper(partial(F.normalize, p=2.0, dim=axis, eps=epsilon), inputs, name=name)


def linspace(start, stop, num, name=None, axis=0):
    # return Lambda(partial(torch.linspace, start=start, end=stop, steps=num), name=name)(inputs)
    return torch.linspace(start=start, end=stop, steps=num)


def log(inputs, name=None):
    return wrapper(partial(torch.log), inputs, name=name)


def logical_and(xx, yy, name=None):
    return wrapper(lambda inputs: torch.logical_and(inputs[0], inputs[1]), [xx, yy], name=name)


def logical_or(xx, yy, name=None):
    return wrapper(lambda inputs: torch.logical_or(inputs[0], inputs[1]), [xx, yy], name=name)


def matmul(xx, yy, transpose_a=False, transpose_b=False, name=None):
    return wrapper(lambda inputs: torch.matmul(inputs[0].T if transpose_a else inputs[0], inputs[1].T if transpose_b else inputs[1]), [xx, yy], name=name)


def moments(inputs, axes, shift=None, keepdims=False, name=None):
    return wrapper(partial(torch.var_mean, dim=axes, keepdim=keepdims), inputs, name=name)


def maximum(xx, yy, name=None):
    if isinstance(yy, torch.Tensor):
        return wrapper(lambda inputs: torch.maximum(inputs[0], inputs[1]), [xx, yy], name=name)
    else:  # maximum doesn't support scalar value
        return wrapper(lambda inputs: torch.clip(xx, min=yy), [xx, yy], name=name)


def minimum(xx, yy, name=None):
    if isinstance(yy, torch.Tensor):
        return wrapper(lambda inputs: torch.minimum(inputs[0], inputs[1]), [xx, yy], name=name)
    else:  # maximum doesn't support scalar value
        return wrapper(lambda inputs: torch.clip(xx, max=yy), [xx, yy], name=name)


def non_max_suppression_with_scores(boxes, scores, max_output_size=100, iou_threshold=0.5, score_threshold=-math.inf, soft_nms_sigma=0.0, name=None):
    from torchvision.ops import nms, batched_nms

    iou_threshold = soft_nms_sigma * 2 if iou_threshold == 1 else iou_threshold  # [TODO] soft_nms_sigma not supported here
    valid_scores_index = torch.where(scores > score_threshold)[0]
    # batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor>
    nms_index = nms(boxes=boxes[valid_scores_index], scores=scores[valid_scores_index], iou_threshold=iou_threshold)
    actual_index = valid_scores_index[nms_index][:max_output_size]
    return actual_index, scores[actual_index]

    # from tensorflow import image
    #
    # if hasattr(boxes, "detach"):
    #     boxes = boxes.detach().numpy()
    # if hasattr(scores, "detach"):
    #     scores = scores.detach().numpy()
    # rr, nms_scores = image.non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
    # return torch.from_numpy(rr.numpy().astype("int64")), torch.from_numpy(nms_scores.numpy().astype("float32"))


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
    return _ZeroPadding(padding=paddings, mode=mode.lower(), value=constant_values)(inputs)
    # pad, output_shape = [], []
    # for pp, cur_shape in zip(paddings[::-1], inputs.shape):
    #     pad += pp
    #     output_shape.append(cur_shape + pp[0] + pp[1])
    # print(f">>>> {pad = }")
    # return Lambda(partial(F.pad, pad=pad, mode=mode.lower(), value=constant_values), output_shape=output_shape, name=name)(inputs)


def pow(inputs, exponent, name=None):
    return wrapper(partial(torch.pow, exponent=exponent), inputs, name=name)


def range(start, limit=None, delta=1, dtype=None, name="range"):
    if limit is None:
        start, limit = 0, start
    dtype = dtype and (getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    if isinstance(limit, Shape) and (None in limit or -1 in limit):
        return wrapper(lambda limit: torch.arange(start=start, end=limit, step=delta, dtype=dtype), limit, output_shape=[None], name=name)
    else:
        return torch.arange(start=start, end=limit, step=delta, dtype=dtype)


def reduce_max(inputs, axis=None, keepdims=False, name=None):
    # return wrapper(partial(torch.max, dim=axis, keepdim=keepdims), inputs, name=name)
    if axis is None:
        return wrapper(lambda inputs: torch.max(inputs), inputs, name=name)
    else:
        return wrapper(lambda inputs: torch.max(inputs, dim=axis, keepdim=keepdims)[0], inputs, name=name)


def reduce_mean(inputs, axis=None, keepdims=False, name=None):
    axis = () if axis is None else axis
    return wrapper(partial(torch.mean, dim=axis, keepdim=keepdims), inputs, name=name)


def reduce_sum(inputs, axis=None, keepdims=False, name=None):
    axis = () if axis is None else axis
    if isinstance(inputs, (list, tuple)) and axis == 0:
        return Add(name=name)(inputs)
    else:
        # return wrapper(lambda xx: xx.sum(dim=axis, keepdim=keepdims), inputs, name=name)
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
        expand_shape = [1] * len(inputs.shape)
        expand_shape.insert(axis + 1, repeats)
        out_shape = [(-1 if ii is None or ii == -1 else ii * repeats) if dim == axis else ii for dim, ii in enumerate(inputs.shape)]
        # return wrapper(lambda inputs: torch.expand_copy(torch.unsqueeze(inputs, axis + 1), expand_shape).contiguous().view(out_shape), inputs, name=name)
        return wrapper(lambda xx: xx.unsqueeze(axis + 1).repeat(expand_shape).contiguous().view(out_shape), inputs, name=name)
    else:
        return wrapper(partial(torch.repeat_interleave, repeats=repeats, dim=axis), inputs, name=name)
    # expand_shape.insert(axis + 1, repeats)
    # expand_shape = tuple(expand_shape)
    # out_shape = [(-1 if ii is None or ii == -1 else ii * repeats) if dim == axis else ii for dim, ii in enumerate(inputs.shape)]
    # return wrapper(lambda inputs: torch.reshape(torch.expand_copy(torch.unsqueeze(inputs, axis + 1), expand_shape), out_shape), inputs, name=name)


def reshape(inputs, shape, name=None):
    # is_shape_dynamic = isinstance(shape, Shape) and (None in shape[1:] or -1 in shape[1:])
    # is_any_shape_dynamic = any([isinstance(ii, Shape) and (-1 in ii or None in ii) for ii in shape])
    if isinstance(inputs, GraphNode) and (isinstance(shape, Shape) and (None in shape[1:] or -1 in shape[1:])):  # If dynamic
        return _ReshapeDynamic(target_shape=list(shape), name=name)([inputs, shape])
    else:
        # return wrapper(partial(torch.reshape, shape=shape), inputs, name=name)
        return wrapper(lambda inputs: inputs.contiguous().view(shape), inputs, name=name)


def resize(inputs, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    if isinstance(inputs, GraphNode) and (isinstance(size, Shape) and (None in size or -1 in size)):  # If dynamic
        return _ResizeDynamic(method=method, preserve_aspect_ratio=preserve_aspect_ratio, antialias=antialias, name=name)([inputs, size])
    elif isinstance(inputs, GraphNode):
        return Lambda(partial(F.interpolate, size=list(size), mode=method, antialias=antialias), name=name)(inputs)  # [TODO] align_corners
        # return Resize(size=size, mode=method, preserve_aspect_ratio=preserve_aspect_ratio, antialias=antialias, name=name)(inputs)  # [TODO] align_corners
    else:  # called directly
        inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)
        if len(inputs.shape) == 3:
            return F.interpolate(inputs[None], size=list(size), mode=method, antialias=antialias)[0]
        else:
            return F.interpolate(inputs, size=list(size), mode=method, antialias=antialias)


def rsqrt(inputs, name=None):
    return wrapper(torch.rsqrt, inputs, name=name)


def shape(inputs):
    return Shape(inputs) if isinstance(inputs, GraphNode) else inputs.shape
    # return inputs.shape


def sigmoid(inputs, axis=None, name=None):
    return wrapper(torch.sigmoid, inputs, name=name)


def sign(inputs, name=None):
    return wrapper(torch.sign, inputs, name=name)


def sin(inputs, name=None):
    return wrapper(torch.sin, inputs, name=name)


def softmax(inputs, axis=None, name=None):
    return wrapper(partial(torch.softmax, dim=-1 if axis is None else axis), inputs, name=name)


def softplus(inputs, name=None):
    return wrapper(F.softplus, inputs, name=name)


def split(inputs, num_or_size_splits, axis=0, num=None, name=None):
    axis = (len(inputs.shape) + axis) if axis < 0 else axis
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


def square(inputs, name=None):
    return wrapper(torch.square, inputs, name=name)


def squeeze(inputs, axis=None, name=None):
    return wrapper(partial(torch.squeeze, dim=axis), inputs, name=name)


def stack(inputs, axis=0, name=None):
    return wrapper(partial(torch.stack, dim=axis), inputs, name=name)


def tanh(inputs, name=None):
    return wrapper(F.tanh, inputs, name=name)


def tile(inputs, multiples, name=None):
    return wrapper(lambda xx: xx.repeat(multiples), inputs, name=name)


def top_k(inputs, k=1, sorted=True, name=None):
    return wrapper(partial(torch.topk, k=k, sorted=sorted), inputs, name=name)


def transpose(inputs, perm=None, conjugate=False, name=None):
    return wrapper(lambda xx: xx.T, inputs, name=name) if perm is None else wrapper(partial(torch.permute, dims=perm), inputs, name=name)


def unstack(inputs, axis=0, name=None):
    assert inputs.shape[axis] is not None
    axis = len(inputs.shape) + axis if axis < 0 else axis
    output_shape = [[jj for ii, jj in enumerate(inputs.shape) if ii != axis]] * inputs.shape[axis]
    return wrapper(partial(torch.unbind, dim=axis), inputs, output_shape=output_shape, name=name)
    # pre_axis_slice = [slice(None)] * axis
    # return [inputs[tuple([*pre_axis_slice, index])] for index in range(axis_shape)]


def where(condition, x=None, y=None, name=None):
    if x is None and y is None:
        output_shape = [None, len(condition.shape)]
        return wrapper(lambda condition: torch.stack(torch.where(condition), axis=1), condition, output_shape=output_shape, name=name)
    elif isinstance(x, GraphNode) and isinstance(y, GraphNode):
        return wrapper(lambda inputs: torch.where(inputs[0], inputs[1], inputs[2]), [condition, x, y], output_shape=condition.shape, name=name)
    elif isinstance(x, GraphNode):
        return wrapper(lambda inputs: torch.where(inputs[0], inputs[1], y), [condition, x], output_shape=condition.shape, name=name)
    elif isinstance(y, GraphNode):
        return wrapper(lambda inputs: torch.where(inputs[0], x, inputs[1]), [condition, y], output_shape=condition.shape, name=name)
    else:
        return wrapper(lambda condition: torch.where(condition, x, y), condition, output_shape=condition.shape, name=name)
