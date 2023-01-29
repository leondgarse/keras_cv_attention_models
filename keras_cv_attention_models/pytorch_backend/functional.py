import torch
import torch.nn.functional as F
from keras_cv_attention_models.pytorch_backend.layers import Lambda
from functools import partial

def norm(inputs, ord='euclidean', axis=1, keepdims=False, name=None):
    return Lambda(partial(torch.norm, p=2, dim=axis, keepdim=keepdims), name=name)(inputs)

def reduce_mean(inputs, axis=None, keepdims=False, name=None):
    return Lambda(partial(torch.mean, dim=axis, keepdim=keepdims), name=name)(inputs)

def reduce_sum(inputs, axis=None, keepdims=False, name=None):
    return Lambda(partial(torch.sum, dim=axis, keepdim=keepdims), name=name)(inputs)

def expand_dims(inputs, axis, name=None):
    return Lambda(partial(torch.unsqueeze, dim=axis), name=name)(inputs)

def squeeze(inputs, axis, name=None):
    return Lambda(partial(torch.squeeze, dim=axis), name=name)(inputs)

def clip_by_value(inputs, clip_value_min, clip_value_max, name=None):
    return Lambda(partial(torch.clip, min=clip_value_min, max=clip_value_max), name=name)(inputs)

def sqrt(inputs, name=None):
    return Lambda(torch.sqrt, name=name)(inputs)

def relu6(inputs, name=None):
    return Lambda(F.relu6, name=name)(inputs)

def tanh(inputs, name=None):
    return Lambda(F.tanh, name=name)(inputs)

def softplus(inputs, name=None):
    return Lambda(F.softplus, name=name)(inputs)

def softmax(inputs, axis=None, name=None):
    return Lambda(partial(F.softmax, dim=axis), name=name)(inputs)

def gelu(inputs, approximate=False, name=None):
    return Lambda(partial(F.gelu, approximate="tanh" if approximate else "none"), name=name)(inputs)

def reshape(inputs, shape, name=None):
    return Lambda(partial(torch.reshape, shape=shape), name=name)(inputs)

def convert_to_tensor(inputs):
    return torch.Tensor(inputs)
