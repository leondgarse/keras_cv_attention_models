import torch
import numpy as np
from torch import nn
from functools import partial


# [TODO] Identity, ConvTranspose2d, Conv2D SAME padding, initializer, MultiHeadAttention, Dropout


""" Basic Layers """

def get_perm(total_axis, from_axis, to_axis):
    # total_axis, from_axis, to_axis = 4, 1, 3 -> [0, 2, 3, 1]
    # total_axis, from_axis, to_axis = 4, 3, 1 -> [0, 3, 1, 2]
    from_axis = (total_axis + from_axis) if from_axis < 0 else from_axis
    to_axis = (total_axis + to_axis) if to_axis < 0 else to_axis
    aa = [ii for ii in range(total_axis) if ii != from_axis]
    aa.insert(to_axis, from_axis)
    return aa


class Weight:
    def __init__(self, name, value):
        self.name, self.shape, self.__value__ = name, value.shape, value

    def __repr__(self):
        return "{} shape={}".format(self.name, self.shape)

    def value(self):
        return self.__value__

    def numpy(self):
        return self.__value__.detach().cpu().numpy()


class GraphNode:
    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, shape, name=None):
        self.shape = shape
        self.name = "graphnode_{}".format(self.num_instances) if name is None else name
        self.pre_nodes, self.pre_node_names, self.next_nodes, self.next_node_names = [], [], [], []
        self.module = lambda xx: xx
        self.__count__()

    def __repr__(self):
        # return ",".join(for kk, vv in zip())
        rr = "{}:\n  in: {}".format(self.name, {ii.name: ii.shape for ii in self.pre_nodes})
        if hasattr(self, "layer") and hasattr(self.layer, "output_shape"):
            rr += "\n  out: {}".format(self.layer.output_shape)
        return rr

    def __getitem__(self, index_expr):
        # print(index_expr)
        # return Slice(index_expr)(self)
        index_expr = index_expr if isinstance(index_expr, (int, slice)) else tuple(index_expr)
        return Lambda(lambda inputs: inputs[index_expr])(self)

    def __add__(self, another):
        return Add()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx + another)(self)

    def __sub__(self, another):
        return Subtract()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx - another)(self)

    def __mul__(self, another):
        return Multiply()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx * another)(self)

    def __truediv__(self, another):
        return Divide()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx / another)(self)

    def __matmul__(self, another):
        # return Matmul()([self, another])
        return Lambda(lambda inputs: torch.matmul(inputs[0], inputs[1]))([self, another])

    def set_pre_nodes(self, pre_nodes):
        pre_nodes = [ii for ii in pre_nodes if isinstance(ii, GraphNode)] if isinstance(pre_nodes, (list, tuple)) else [pre_nodes]
        self.pre_nodes += pre_nodes
        self.pre_node_names += [ii.name for ii in pre_nodes]

    def set_next_nodes(self, next_nodes):
        next_nodes = next_nodes if isinstance(next_nodes, (list, tuple)) else [next_nodes]
        self.next_nodes += next_nodes
        self.next_node_names += [ii.name for ii in next_nodes]


class Input(GraphNode):
    def __init__(self, shape, name=None):
        shape = [None, *shape]
        name = "input_{}".format(self.num_instances) if name is None else name
        super().__init__(shape, name=name)


class Layer(nn.Module):
    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name, self.kwargs = self.verify_name(name), kwargs
        self.built = False
        self.use_layer_as_module = False  # True if called add_weights, or typical keras layers with overwriting call function
        if not hasattr(self, "module"):
            self.module = lambda xx: xx
        self.__count__()

    def build(self, input_shape: torch.Size):
        self.input_shape = input_shape
        self.__output_shape__ = self.compute_output_shape(input_shape)
        # if hasattr(self, "call"):  # General keras layers with call function
        #     self.forward = self.call
        #     self.call = self.call
        self.built = True

    def call(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    def forward(self, inputs, **kwargs):
        if not self.built:
            self.build([() if isinstance(ii, (int, float)) else ii.shape for ii in inputs] if isinstance(inputs, (list, tuple)) else inputs.shape)
        if isinstance(inputs, GraphNode) or (isinstance(inputs, (list, tuple)) and any([isinstance(ii, GraphNode) for ii in inputs])):
            output_shape = self.compute_output_shape(self.input_shape)
            # if isinstance(output_shape[0], (list, tuple))
            cur_node = GraphNode(output_shape, name=self.name)
            if self.use_layer_as_module:  # General keras layers with call function, mostly own weights
                cur_node.callable = self
            else:
                cur_node.callable = self.module if len(kwargs) == 0 else partial(self.module, **kwargs)
            cur_node.layer = self
            cur_node.set_pre_nodes(inputs)

            inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
            for ii in inputs:
                if isinstance(ii, GraphNode):
                    ii.set_next_nodes(cur_node)
            self.output = self.node = cur_node
            return cur_node
        else:
            return self.call(inputs, **kwargs)

    @property
    def weights(self):
        return [Weight(name=self.name + "/" + kk.split(".")[-1], value=vv) for kk, vv in self.state_dict().items() if not kk.endswith(".num_batches_tracked")]

    @property
    def trainable_weights(self):
        return self.weights

    @property
    def non_trainable_weights(self):
        return []

    def add_weight(self, name=None, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None):
        self.use_layer_as_module = True
        if isinstance(initializer, str):
            initializer = torch.ones if initializer == "ones" else torch.zeros
        return nn.Parameter(initializer(shape), requires_grad=trainable)

    def get_weights(self):
        return [ii.value().detach().cpu().numpy() for ii in self.weights]

    def set_weights(self, weights):
        return self.load_state_dict({kk: torch.from_numpy(vv) for kk, vv in zip(self.state_dict().keys(), weights)})

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def output_shape(self):
        return getattr(self, "__output_shape__", None)

    def verify_name(self, name):
        return "{}_{}".format(self.__class__.__name__.lower(), self.num_instances) if name == None else name

    def get_config(self):
        config = {"name": self.name}
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Lambda(Layer):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.module = func

    def get_config(self):
        config = super().get_config()
        config.update({"func": self.module})
        return config

    def compute_output_shape(self, input_shape):
        # print(self.module, input_shape)
        input_shapes = [input_shape] if isinstance(input_shape[0], int) or input_shape[0] is None else input_shape
        inputs = [torch.ones([0 if ii is None or ii == -1 else ii for ii in input_shape]) for input_shape in input_shapes]  # Regards 0 as dynamic shape
        output_shape = list(self.module(inputs[0]).shape) if len(inputs) == 1 else list(self.module(inputs).shape)
        return [None if ii == 0 else ii for ii in output_shape]  # Regards 0 as dynamic shape


class Shape(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = lambda inputs: inputs.shape
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)


class Activation(Layer):
    def __init__(self, activation=None, **kwargs):
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.activation is None or self.activation == "linear":
            self.module = torch.nn.Identity()
        elif isinstance(self.activation, str) and self.activation == "softmax":
            self.module = partial(torch.softmax, dim=1)
        elif isinstance(self.activation, str) and self.activation == "swish":
            self.module = torch.nn.SiLU()
        elif isinstance(self.activation, str):
            self.module = getattr(torch.functional.F, self.activation)
        else:
            self.module = self.activation
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"activation": self.activation})
        return config


""" Merge Layers """


class _Merge(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.check_input(input_shape)
        self.module = self.merge_function
        super().build(input_shape)

    def check_input(self, input_shape):
        output_shape = self.compute_output_shape(input_shape)
        if len(output_shape) == 0:
            return

        valid_input_shape = [ii for ii in input_shape if len(ii) != 0]  # exclude scalar values
        max_dim = len(output_shape)
        result = not any([any([ii[dim] != 1 and output_shape[dim] != ii[dim] for dim in range(1, max_dim)]) for ii in valid_input_shape])
        assert result, "input_shapes not all equal: {}".format(input_shape)

    def compute_output_shape(self, input_shape):
        valid_input_shape = [ii[1:] for ii in input_shape if len(ii) != 0]  # exclude scalar values, also cut batch dimension
        if len(valid_input_shape) == 0:  # Should not happen...
            return ()

        max_dim = max([len(ii) for ii in valid_input_shape])
        valid_input_shape = np.array([[1] * (max_dim - len(ii)) + list(ii) for ii in valid_input_shape])  # expands 1 on the head
        max_shape = valid_input_shape.max(0)
        return [None, *max_shape]


class Add(_Merge):
    def merge_function(self, inputs):
        output = torch.add(inputs[0], inputs[1])
        for ii in inputs[2:]:
            output = torch.add(output, ii)
        return output


class Divide(_Merge):
    def merge_function(self, inputs):
        output = torch.divide(inputs[0], inputs[1])
        for ii in inputs[2:]:
            output = torch.divide(output, ii)
        return output


class Subtract(_Merge):
    def merge_function(self, inputs):
        output = torch.subtract(inputs[0], inputs[1])
        for ii in inputs[2:]:
            output = torch.subtract(output, ii)
        return output


class Matmul(_Merge):
    def merge_function(self, inputs):
        output = torch.matmul(inputs[0], inputs[1])
        for ii in inputs[2:]:
            output = torch.matmul(output, ii)
        return output

    def check_input(self, input_shape):
        base, dims = input_shape[0], len(input_shape[0])
        result = not any([any([base[dim] != ii[dim] for dim in range(1, dims) if dim != self.axis]) for ii in input_shape[1:]])
        assert result, "input_shapes excpets concat axis {} not all equal: {}".format(self.axis, input_shape)

    def compute_output_shape(self, input_shape):
        dims = len(input_shape[0])
        return [sum([ii[dim] for ii in input_shape]) if dim == self.axis else input_shape[0][dim] for dim in range(dims)]


class Multiply(_Merge):
    def merge_function(self, inputs):
        output = torch.multiply(inputs[0], inputs[1])
        for ii in inputs[2:]:
            output = torch.multiply(output, ii)
        return output


class Concatenate(_Merge):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def merge_function(self, inputs):
        return torch.concat(inputs, dim=self.axis)

    def check_input(self, input_shape):
        base, dims = input_shape[0], len(input_shape[0])
        result = not any([any([base[dim] != ii[dim] for dim in range(1, dims) if dim != self.axis]) for ii in input_shape[1:]])
        assert result, "input_shapes excpets concat axis {} not all equal: {}".format(self.axis, input_shape)

    def compute_output_shape(self, input_shape):
        dims = len(input_shape[0])
        return [sum([ii[dim] for ii in input_shape]) if dim == self.axis else input_shape[0][dim] for dim in range(dims)]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


""" Layers with weights """


class BatchNormalization(Layer):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, gamma_initializer="ones", **kwargs):
        self.axis, self.momentum, self.epsilon, self.center, self.gamma_initializer = axis, momentum, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        module = nn.BatchNorm2d(num_features=input_shape[self.axis], eps=self.epsilon, momentum=1 - self.momentum, affine=self.center)
        if self.axis == 1:
            self.module = module
        else:
            ndims = len(input_shape)
            perm = get_perm(total_axis=ndims, from_axis=self.axis, to_axis=1)  # like axis=-1 -> [0, 3, 1, 2]
            revert_perm = get_perm(total_axis=ndims, from_axis=1, to_axis=self.axis)  # like axis=-1 -> [0, 2, 3, 1]
            self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "momentum": self.momentum, "epsilon": self.epsilon, "center": self.center})
        return config


class Conv(Layer):
    def __init__(
        self, filters, kernel_size=1, strides=1, padding="VALID", dilation_rate=1, use_bias=True, groups=1, kernel_initializer="glorot_uniform", **kwargs
    ):
        self.filters, self.padding, self.use_bias, self.groups, self.kernel_initializer = filters, padding, use_bias, groups, kernel_initializer
        self.kernel_size, self.dilation_rate, self.strides = kernel_size, dilation_rate, strides
        self.module_class = None  # Auto set by len(input_shape)
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_dims = len(input_shape) - 2  # Conv2D -> 2, Conv1D -> 1
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, (list, tuple)) else [self.kernel_size] * num_dims
        self.dilation_rate = self.dilation_rate if isinstance(self.dilation_rate, (list, tuple)) else [self.dilation_rate] * num_dims
        self.strides = self.strides if isinstance(self.strides, (list, tuple)) else [self.strides] * num_dims
        self.filters = self.filters if self.filters > 0 else input_shape[1]  # In case DepthwiseConv2D

        if isinstance(self.padding, str):
            self._pad = [ii // 2 for ii in self.kernel_size] if self.padding.upper() == "SAME" else [0] * num_dims
        else:  # int or list or tuple with specific value
            self._pad = padding if isinstance(padding, (list, tuple)) else [padding] * num_dims

        if self.module_class is None:
            self.module_class = nn.Conv1d if len(input_shape) == 3 else (nn.Conv2d if len(input_shape) == 4 else nn.Conv3d)

        self.module = self.module_class(
            in_channels=input_shape[1],
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._pad,
            dilation=self.dilation_rate,
            groups=self.groups,
            bias=self.use_bias,
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        size = input_shape[2:]
        dilated_filter_size = [kk + (kk - 1) * (dd - 1) for kk, dd in zip(self.kernel_size, self.dilation_rate)]
        if self.padding.upper() == "VALID":
            size = [ii - jj + 1 for ii, jj in zip(size, dilated_filter_size)]
        size = [(ii + jj - 1) // jj for ii, jj in zip(size, self.strides)]
        return [None, self.filters, *size]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "groups": self.groups,
            }
        )
        return config


class Conv1D(Conv):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (2, 1, 0))
        return self.set_weights(weights)


class Conv2D(Conv):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
        return self.set_weights(weights)


class DepthwiseConv2D(Conv):
    def __init__(self, kernel_size=1, strides=1, padding="VALID", dilation_rate=(1, 1), use_bias=True, kernel_initializer="glorot_uniform", **kwargs):
        filters, groups = -1, -1
        super().__init__(filters, kernel_size, strides, padding, dilation_rate, use_bias, groups, kernel_initializer, **kwargs)

    def build(self, input_shape):
        self.groups = input_shape[1]
        super().build(input_shape)

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
        return self.set_weights(weights)


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, axis=-1, kernel_initializer="glorot_uniform", **kwargs):
        self.units, self.activation, self.use_bias, self.axis, self.kernel_initializer = units, activation, use_bias, axis, kernel_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        module = nn.Linear(in_features=input_shape[self.axis], out_features=self.units, bias=self.use_bias)
        if self.axis == len(input_shape) - 1:
            self.module = module if self.activation is None else nn.Sequential(module, Activation(self.activation))
        else:
            ndims = len(input_shape)
            perm = get_perm(total_axis=ndims, from_axis=self.axis, to_axis=-1)  # like axis=1 -> [0, 2, 3, 1]
            revert_perm = get_perm(total_axis=ndims, from_axis=-1, to_axis=self.axis)  # like axis=1 -> [0, 3, 1, 2]
            if self.activation is None:
                self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
            else:
                self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]), Activation(self.activation))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        return input_shape[: self.axis] + [self.units] + input_shape[self.axis + 1 :]

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0])
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0])
        return self.set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation, "use_bias": self.use_bias, "axis": self.axis})
        return config


class GroupNormalization(Layer):
    def __init__(self, groups=32, axis=1, epsilon=0.001, center=True, gamma_initializer="ones", **kwargs):
        self.groups, self.axis, self.epsilon, self.center, self.gamma_initializer = groups, axis, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        module = nn.GroupNorm(num_groups=self.groups, num_channels=input_shape[self.axis], eps=self.epsilon, affine=self.center)
        # Default nn.GroupNorm is apllied on first non-batch dimension
        if self.axis == 1:
            self.module = module
        else:
            ndims = len(input_shape)
            perm = get_perm(total_axis=ndims, from_axis=self.axis, to_axis=1)  # like axis=-1 -> [0, 3, 1, 2]
            revert_perm = get_perm(total_axis=ndims, from_axis=1, to_axis=self.axis)  # like axis=-1 -> [0, 2, 3, 1]
            self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups, "axis": self.axis, "epsilon": self.epsilon, "center": self.center})
        return config


class LayerNormGeneral(nn.Module):
    def __init__(self, normalized_shape, normalized_dim=(-1, ), scale=True, bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim, self.use_scale, self.use_bias, self.eps = normalized_dim, scale, bias, eps
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, inputs):
        center = inputs - inputs.mean(self.normalized_dim, keepdim=True)
        scale = center.pow(2).mean(self.normalized_dim, keepdim=True)
        inputs = center / torch.sqrt(scale + self.eps)
        if self.use_scale:
            inputs = inputs * self.weight
        if self.use_bias:
            inputs = inputs + self.bias
        return inputs


class LayerNormalization(Layer):
    def __init__(self, axis=1, epsilon=1e-5, center=True, gamma_initializer="ones", **kwargs):
        self.axis, self.epsilon, self.center, self.gamma_initializer = axis, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        if self.center:
            module = nn.LayerNorm(normalized_shape=input_shape[self.axis], eps=self.epsilon)
            # Default nn.LayerNorm is apllied on last dimension
            if self.axis == len(input_shape) - 1:
                self.module = module
            else:
                ndims = len(input_shape)
                perm = get_perm(total_axis=ndims, from_axis=self.axis, to_axis=-1)  # like axis=1 -> [0, 2, 3, 1]
                revert_perm = get_perm(total_axis=ndims, from_axis=-1, to_axis=self.axis)  # like axis=1 -> [0, 3, 1, 2]
                self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
        else:
            self.module = LayerNormGeneral(normalized_shape=input_shape[self.axis], normalized_dim=self.axis, eps=self.epsilon, bias=self.center)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon, "center": self.center})
        return config


class PReLU(Layer):
    def __init__(self, alpha_initializer="zeros", alpha_regularizer=None, alpha_constraint=None, shared_axes=None, **kwargs):
        self.shared_axes = shared_axes
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.PReLU(num_parameters=input_shape[1], init=0.25)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"shared_axes": self.shared_axes})
        return config

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.expand_dims(np.expand_dims(weights[0], 0), 0)
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.squeeze(weights[0])
        return self.set_weights(weights)


""" Layers with no weights """


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    from drop_path https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/drop.py#L137
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
        super().__init__()

    def forward(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = inputs.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return inputs * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Dropout(Layer):
    def __init__(self, rate, noise_shape=None, **kwargs):
        self.rate, self.noise_shape = rate, noise_shape
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.noise_shape is None:
            dims, rate = len(input_shape), self.rate
            self.module = torch.nn.Dropout1d(p=rate) if dims == 3 else (torch.nn.Dropout2d(p=rate) if dims == 4 else torch.nn.Dropout3d(p=rate))
        else:
            self.module = DropPath(drop_prob=self.rate)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "noise_shape": self.noise_shape})
        return config


class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.Flatten()
        super().build(input_shape)

    def compute_output_shape(self, inptut_shape):
        return [None, int(np.prod(inptut_shape[1:]))]


class Pooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", reduce="mean", **kwargs):
        self.pool_size, self.strides, self.padding, self.reduce = pool_size, strides, padding, reduce
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else [pool_size, pool_size]
        self.strides = strides if isinstance(strides, (list, tuple)) else [strides, strides]
        super().__init__(**kwargs)

    def build(self, input_shape):
        pool_size = self.pool_size
        if isinstance(self.padding, str):
            pad = (pool_size[0] // 2, pool_size[1] // 2) if self.padding.upper() == "SAME" else (0, 0)
        else:  # int or list or tuple with specific value
            pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
        self._pad = pad

        if self.reduce.lower() == "max":
            self.module = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.strides, padding=pad)
        else:
            self.module = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.strides, padding=pad)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, channels, height, width = input_shape
        height = (height + 2 * self._pad[0] - (self.pool_size[0] - self.strides[0])) // self.strides[0]  # Not considering dilation
        width = (width + 2 * self._pad[1] - (self.pool_size[1] - self.strides[1])) // self.strides[1]  # Not considering dilation
        return [batch, channels, height, width]

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding, "reduce": self.reduce})
        return config


class AvgPool2D(Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(reduce="mean", **kwargs)


class MaxPool2D(Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(reduce="max", **kwargs)


class GlobalAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(1))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:2]


class GlobalAveragePooling2D(Layer):
    def __init__(self, keepdims=False, **kwargs):
        self.keepdims = keepdims
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = torch.nn.AdaptiveAvgPool2d(1) if self.keepdims else torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[:2] + [1] * (len(input_shape) - 2)) if self.keepdims else input_shape[:2]

    def get_config(self):
        config = super().get_config()
        config.update({"keepdims": self.keepdims})
        return config


class Permute(Layer):
    def __init__(self, dims, **kwargs):
        self.dims = dims
        super().__init__(**kwargs)
        assert sorted(dims) == list(range(1, len(dims) + 1)), "The set of indices in `dims` must be consecutive and start from 1. dims: {}".format(dims)

    def build(self, input_shape):
        self.module = partial(torch.permute, dims=[0, *self.dims])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # output_shape = input_shape.copy()
        # for i, dim in enumerate(self.dims):
        #     output_shape[i + 1] = input_shape[dim]
        return [input_shape[0]] + [input_shape[dim] for dim in self.dims]

    def get_config(self):
        config = super().get_config()
        config.update({"dims": self.dims})
        return config


class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        self.target_shape = [-1 if ii is None else ii for ii in target_shape]
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_unknown_dim = sum([ii == -1 for ii in self.target_shape])
        assert num_unknown_dim < 2, "At most one unknown dimension in output_shape: {}".format(self.target_shape)

        total_size = np.prod(input_shape[1:])
        if num_unknown_dim == 1:
            unknown_dim = total_size // (-1 * np.prod(self.target_shape))
            self.target_shape = [unknown_dim if ii == -1 else ii for ii in self.target_shape]
        assert total_size == np.prod(self.target_shape), "Total size of new array must be unchanged, {} -> {}".format(input_shape, self.target_shape)

        self.module = partial(torch.reshape, shape=[-1, *self.target_shape])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], *self.target_shape]

    def get_config(self):
        config = super().get_config()
        config.update({"target_shape": self.target_shape})
        return config


class Softmax(Layer):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = partial(torch.softmax, dim=self.axis)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ZeroPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        assert len(padding) == 2 if isinstance(padding, (list, tuple)) else isinstance(padding, int), "padding should be 2 values or an int: {}".format(padding)
        self.padding = list(padding) if isinstance(padding, (list, tuple)) else [padding, padding]
        super().__init__(**kwargs)

    def build(self, input_shape):
        padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]  # [left, right, top, bottom]
        self.module = torch.nn.ZeroPad2d(padding=padding)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2] + self.padding[0] * 2, input_shape[3] + self.padding[1] * 2]

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config
