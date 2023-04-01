import inspect
import torch
import numpy as np
from torch import nn
from functools import partial
from keras_cv_attention_models.pytorch_backend import initializers


# [TODO] Identity, Normalization, Rescaling, SeparableConv2D


""" Basic Layers """


def to_tuple(param, num_dims=2):
    return tuple(param if isinstance(param, (list, tuple)) else [param] * num_dims)


def get_perm(total_axis, from_axis, to_axis):
    # total_axis, from_axis, to_axis = 4, 1, 3 -> [0, 2, 3, 1]
    # total_axis, from_axis, to_axis = 4, 3, 1 -> [0, 3, 1, 2]
    from_axis = (total_axis + from_axis) if from_axis < 0 else from_axis
    to_axis = (total_axis + to_axis) if to_axis < 0 else to_axis
    aa = [ii for ii in range(total_axis) if ii != from_axis]
    aa.insert(to_axis, from_axis)
    return aa


def tf_same_pad(size, kernel_size, stride, dilation_rate=1):
    if size is None:
        return kernel_size - stride  # Regarding as size % stride == 0
    else:
        return max((np.math.ceil(size / stride) - 1) * stride + (kernel_size - 1) * dilation_rate + 1 - size, 0)


def compute_conv_output_size(input_shape, kernel_size, strides=1, padding="valid", dilation_rate=1):
    size = input_shape if len(input_shape) == 2 else input_shape[2:]
    kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size] * len(size)
    dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else [dilation_rate] * len(size)
    strides = strides if isinstance(strides, (list, tuple)) else [strides] * len(size)
    dilated_filter_size = [kk + (kk - 1) * (dd - 1) for kk, dd in zip(kernel_size, dilation_rate)]
    if (isinstance(padding, str) and padding.upper() == "VALID") or (isinstance(padding, (list, tuple)) and max(padding) == 0) or padding == 0:
        size = [None if ii is None else (ii - jj + 1) for ii, jj in zip(size, dilated_filter_size)]
    size = [None if ii is None else ((ii + jj - 1) // jj) for ii, jj in zip(size, strides)]
    return size


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
        rr = "{}:\n  nodes in: {}".format(self.name, {ii.name: ii.shape for ii in self.pre_nodes})
        rr = "{}:\n  nodes out: {}".format(self.name, {ii.name: ii.shape for ii in self.next_nodes})
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

    def __radd__(self, another):
        return self.__add__(another)

    def __sub__(self, another):
        return Subtract()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx - another)(self)

    def __rsub__(self, another):
        return self.__sub__(another)

    def __mul__(self, another):
        return Multiply()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx * another)(self)

    def __rmul__(self, another):
        return self.__mul__(another)

    def __truediv__(self, another):
        return Divide()([self, another]) if isinstance(another, GraphNode) else Lambda(lambda xx: xx / another)(self)

    def __rtruediv__(self, another):
        return self.__truediv__(another)

    # def __floordiv__(self, another):
    #     return Lambda(lambda xx: xx // another)(self)

    def __matmul__(self, another):
        # return Matmul()([self, another])
        return Lambda(lambda inputs: torch.matmul(inputs[0], inputs[1]))([self, another])

    def __pow__(self, exponent):
        # return Lambda(lambda inputs: torch.pow(inputs, exponent))(self)
        return Lambda(partial(torch.pow, exponent=exponent))(self)

    def set_shape(self, shape):
        self.shape = shape

    def set_pre_nodes(self, pre_nodes):
        pre_nodes = [ii for ii in pre_nodes if isinstance(ii, GraphNode)] if isinstance(pre_nodes, (list, tuple)) else [pre_nodes]
        self.pre_nodes += pre_nodes
        self.pre_node_names += [ii.name for ii in pre_nodes]

    def set_next_nodes(self, next_nodes):
        next_nodes = next_nodes if isinstance(next_nodes, (list, tuple)) else [next_nodes]
        self.next_nodes += next_nodes
        self.next_node_names += [ii.name for ii in next_nodes]


class Shape(GraphNode):
    """
    >>> from keras_cv_attention_models.pytorch_backend import layers, functional
    >>> aa = layers.Input([4, 4, 32])
    >>> bb = functional.shape(aa)
    >>> print(bb)
    >>> # [-1, 4, 4, 32]
    >>> aa.set_shape([None, 3, 4, 5])
    >>> print(bb)
    >>> # [-1, 3, 4, 5]
    """

    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, input_node, index_expr=slice(None), name=None):
        self.name = "shape_{}".format(self.num_instances) if name is None else name
        self.input_node, self.index_expr = input_node, index_expr
        self.is_slice = not isinstance(self.index_expr, int)
        self.shape = (self.__len__(),) if self.is_slice else ()
        self.build()
        self.__count__()

    def build(self):
        if self.is_slice:
            self.callable = lambda inputs: [-1 if ii is None else ii for ii in inputs.shape[self.index_expr]]
        else:
            self.callable = lambda inputs: inputs.shape[self.index_expr]
        self.layer = self
        self.pre_nodes = [self.input_node]
        self.pre_node_names = [self.input_node.name]
        self.input_node.set_next_nodes(self)
        self.weights = []
        self.next_nodes, self.next_node_names = [], []

    def set_next_nodes(self, next_nodes):
        next_nodes = next_nodes if isinstance(next_nodes, (list, tuple)) else [next_nodes]
        self.next_nodes += next_nodes
        self.next_node_names += [ii.name for ii in next_nodes]

    def __getitem__(self, index_expr):
        return Shape(self.input_node, index_expr)

    def __index__(self):
        return self.input_node.shape[self.index_expr]

    def __value__(self):
        return [-1 if ii is None else ii for ii in self.input_node.shape[self.index_expr]] if self.is_slice else self.input_node.shape[self.index_expr]

    def __len__(self):
        return len(self.__value__()) if self.is_slice else 1

    def __int__(self):
        return int(self.__value__())

    def __iter__(self):
        return (ii for ii in self.__value__())

    def __repr__(self):
        return str(self.__value__())


class Input(GraphNode):
    def __init__(self, shape, name=None, dtype=None):
        shape = [None, *shape]
        name = "input_{}".format(self.num_instances) if name is None else name
        super().__init__(shape, name=name)
        self.dtype = dtype


class Layer(nn.Module):
    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name, self.kwargs = self.verify_name(name), kwargs
        self.built = False
        # self.is_graph_node_input = False
        self.use_layer_as_module = False  # True if called add_weights, or typical keras layers with overwriting call function
        if not hasattr(self, "module"):
            self.module = lambda xx: xx
        self.__count__()
        self.outputs = self.nodes = None

    def build(self, input_shape: torch.Size):
        # pass
        # self.input_shape = input_shape
        # self.dtype = self.module.weight.dtype if hasattr(self.module, "weight") else "float32"
        # if self.is_graph_node_input:  # When exporting onnx/pth, model will be built again, and may throws error in compute_output_shape with 0 input_shape
        # self.__output_shape__ = self.compute_output_shape(input_shape)
        # if hasattr(self, "call"):  # General keras layers with call function
        #     self.forward = self.call
        #     self.call = self.call
        self.built = True

    def call(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    def forward(self, inputs, **kwargs):
        # self.is_graph_node_input = isinstance(inputs, GraphNode) or (isinstance(inputs, (list, tuple)) and any([isinstance(ii, GraphNode) for ii in inputs]))
        if not self.built:
            self.build([() if isinstance(ii, (int, float)) else ii.shape for ii in inputs] if isinstance(inputs, (list, tuple)) else inputs.shape)
            self.built = True

        if isinstance(inputs, GraphNode) or (isinstance(inputs, (list, tuple)) and any([isinstance(ii, GraphNode) for ii in inputs])):
            self.input_shape = [() if isinstance(ii, (int, float)) else ii.shape for ii in inputs] if isinstance(inputs, (list, tuple)) else inputs.shape
            self.__output_shape__ = self.compute_output_shape(self.input_shape)

            # output_shape = self.compute_output_shape(input_shape)
            # if isinstance(self.__output_shape__[0], (list, tuple))
            cur_node = GraphNode(self.__output_shape__, name=self.name if self.nodes is None else (self.name + "_{}".format(len(self.nodes))))
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

            if self.nodes is None:
                self.outputs = self.nodes = [cur_node]
                self.output = self.node = cur_node
            else:
                self.nodes.append(cur_node)
                self.outputs.append(cur_node)
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

    def add_weight(self, name=None, shape=None, dtype="float32", initializer="zeros", regularizer=None, trainable=True):
        self.use_layer_as_module = True
        initializer = getattr(initializers, initializer)() if isinstance(initializer, str) else initializer
        param = nn.Parameter(initializer(shape, dtype=dtype), requires_grad=trainable)
        return param

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

    def extra_repr(self):
        config = self.get_config()
        # config.pop("name")
        return ", ".join(["{}={}".format(kk, vv) for kk, vv in config.items()])


class Lambda(Layer):
    def __init__(self, func, **kwargs):
        self.module = func
        super().__init__(**kwargs)

    def build(self, input_shape: torch.Size):
        super().build(input_shape)
        # self.forward = self.module

    def get_config(self):
        config = super().get_config()
        config.update({"func": self.module})
        return config

    def compute_output_shape(self, input_shape):
        # print(self.module, input_shape)
        input_shapes = [input_shape] if isinstance(input_shape[0], int) or input_shape[0] is None else input_shape
        # print(input_shapes)
        inputs = [torch.ones([0 if ii is None or ii == -1 else ii for ii in input_shape]) for input_shape in input_shapes]  # Regards 0 as dynamic shape
        output_shape = list(self.module(inputs[0]).shape) if len(inputs) == 1 else list(self.module(inputs).shape)
        return [None if ii == 0 else ii for ii in output_shape]  # Regards 0 as dynamic shape

    def extra_repr(self):
        from types import LambdaType

        if isinstance(self.module, partial) and isinstance(self.module.func, LambdaType):
            func_str = "{}, {}".format(inspect.getsource(self.module.func), self.module.keywords)
        if isinstance(self.module, partial):
            func_str = "{}, {}".format(self.module.func.__name__, self.module.keywords)
        else:
            func_str = inspect.getsource(self.module)
        return "name: " + self.name + ", " + func_str.strip()


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
            self.module = getattr(torch, self.activation, getattr(torch.functional.F, self.activation))
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

    def preprocess_input_shape(self, input_shape):
        valid_input_shape = [ii for ii in input_shape if len(ii) != 0]  # exclude scalar values, also cut batch dimension
        if len(valid_input_shape) == 0:  # Should not happen...
            return (), ()

        max_dim = max([len(ii) for ii in valid_input_shape]) - 1
        valid_input_shape = [ii if len(ii) == 1 else ii[1:] for ii in valid_input_shape]  # Regard None as 0
        valid_input_shape = [[1] * (max_dim - len(ii)) + list(ii) for ii in valid_input_shape]  # expands 1 on the head
        valid_input_shape = np.array([[np.inf if jj is None else jj for jj in ii] for ii in valid_input_shape])  # Regard None as inf
        max_shape = valid_input_shape.max(0)
        max_shape = [None if np.isinf(ii) else int(ii) for ii in max_shape]  # Regard inf as None
        return valid_input_shape, max_shape

    def check_input(self, input_shape):
        valid_input_shape, max_shape = self.preprocess_input_shape(input_shape)
        if len(valid_input_shape) == 0:
            return
        check_single_func = lambda ii: all([ss is None or tt is None or ss == 1 or ss == tt for ss, tt in zip(ii, max_shape)])
        assert all([check_single_func(ii) for ii in valid_input_shape]), "input_shapes not all equal: {}".format(input_shape)

    def compute_output_shape(self, input_shape):
        valid_input_shape, max_shape = self.preprocess_input_shape(input_shape)
        return () if len(valid_input_shape) == 0 else [None, *max_shape]


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
        axis = (dims + self.axis) if self.axis < 0 else self.axis
        result = not any([any([base[dim] != ii[dim] for dim in range(1, dims) if dim != axis]) for ii in input_shape[1:]])
        assert result, "input_shapes excpets concat axis {} not all equal: {}".format(self.axis, input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0]).copy()
        combine_dims = [ii[self.axis] for ii in input_shape]
        output_shape[self.axis] = None if any([ii is None for ii in combine_dims]) else sum(combine_dims)
        return output_shape
        # dims = len(input_shape[0])
        # axis = (dims + self.axis) if self.axis < 0 else self.axis
        # return [sum([ii[dim] for ii in input_shape]) if dim == axis else input_shape[0][dim] for dim in range(dims)]

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
        gamma_initializer = getattr(initializers, self.gamma_initializer)() if isinstance(self.gamma_initializer, str) else self.gamma_initializer
        module.weight.data = gamma_initializer(list(module.weight.shape))  # not using gamma_initializer(module.weight) for compiling with TF

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


class _SamePadding(nn.Module):
    """Perform SAME padding like TF"""

    def __init__(self, kernel_size, strides, dilation_rate=1, ndims=2):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size] * ndims
        self.strides = strides if isinstance(strides, (list, tuple)) else [strides] * ndims
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else [dilation_rate] * ndims
        self.ndims = len(self.kernel_size)
        self.fixed_padding = None

    def build_pad(self, input_shape):
        pad = [tf_same_pad(ii, kk, ss, dd) for ii, kk, ss, dd in zip(input_shape[2:], self.kernel_size, self.strides, self.dilation_rate)]
        padding = []
        for pp in pad[::-1]:
            padding += [pp // 2, pp - pp // 2]
        padding += [0, 0, 0, 0]
        return padding

    def forward(self, inputs):
        padding = self.build_pad(inputs.shape) if self.fixed_padding is None else self.fixed_padding
        return torch.functional.F.pad(inputs, pad=padding)


class _ZeroPadding(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = padding
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == len(self.padding), "padding shoule be same length with input, including batch dimension, padding: {}".format(self.padding)
        padding = []
        for pad in self.padding[::-1]:
            assert len(pad) == 2, "each element in padding should be exactly 2 values, padding: {}".format(self.padding)
            padding += pad

        self.module = partial(torch.functional.F.pad, pad=padding)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [None if ii is None else ii + pp[0] + pp[1] for ii, pp in zip(input_shape, self.padding)]

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class _BaseConvPool(Layer):
    def __init__(self, kernel_size=1, strides=1, padding="VALID", dilation_rate=1, **kwargs):
        self.kernel_size, self.dilation_rate, self.strides, self.padding = kernel_size, dilation_rate, strides, padding
        super().__init__(**kwargs)

    def build_module(self, input_shape):
        raise NotImplementedError()

    def build(self, input_shape):
        num_dims = len(input_shape) - 2  # Conv2D -> 2, Conv1D -> 1
        self.kernel_size = to_tuple(self.kernel_size, num_dims=num_dims)
        self.dilation_rate = to_tuple(self.dilation_rate, num_dims=num_dims)
        self.strides = to_tuple(self.strides, num_dims=num_dims)

        # elif isinstance(self.padding, str):
        if isinstance(self.padding, str):
            self._pad = [0] * num_dims  # Alo set to 0 for "SAME" if input_shape[-1] is None, will apply TF like same padding later.
            padding = self.padding.upper()
        else:  # int or list or tuple with specific value
            self._pad = self.padding if isinstance(self.padding, (list, tuple)) else [self.padding] * num_dims
            padding = self.padding

        if None not in input_shape[2:] and padding == "SAME":
            pad = [tf_same_pad(*args) for args in zip(input_shape[2:], self.kernel_size, self.strides, self.dilation_rate)]
            half_pad = [pp // 2 for pp in pad]
            if all([ii == 2 * jj for ii, jj in zip(pad, half_pad)]):
                self._pad = half_pad  # Using module pad directly, avoid applying additional `_SamePadding` layer
                padding = "VALID"

        module = self.build_module(input_shape)
        if max(self.kernel_size) > 0 and padding == "SAME":
            # TF like same padding
            same_padding = _SamePadding(kernel_size=self.kernel_size, strides=self.strides, dilation_rate=self.dilation_rate, ndims=num_dims)
            if None not in input_shape[1:]:
                same_padding.fixed_padding = same_padding.build_pad(input_shape)  # Set fixed padding
            self.module = nn.Sequential(same_padding, module)
        else:
            self.module = module
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        output_size = compute_conv_output_size(input_shape, self.kernel_size, self.strides, self.padding, dilation_rate=self.dilation_rate)
        return [None, input_shape[1], *output_size]

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding, "dilation_rate": self.dilation_rate})
        return config


class Conv(_BaseConvPool):
    def __init__(
        self, filters, kernel_size=1, strides=1, padding="VALID", dilation_rate=1, use_bias=True, groups=1, kernel_initializer="glorot_uniform", **kwargs
    ):
        super().__init__(kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, **kwargs)
        self.filters, self.use_bias, self.groups, self.kernel_initializer = filters, use_bias, groups, kernel_initializer
        self.module_class = None  # Auto set by len(input_shape)

    def build_module(self, input_shape):
        self.filters = int(self.filters if self.filters > 0 else input_shape[1])  # In case DepthwiseConv2D
        if self.module_class is None:
            self.module_class = nn.Conv1d if len(input_shape) == 3 else (nn.Conv2d if len(input_shape) == 4 else nn.Conv3d)

        module = self.module_class(
            in_channels=input_shape[1],
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._pad,
            dilation=self.dilation_rate,
            groups=self.groups,
            bias=self.use_bias,
        )
        kernel_initializer = getattr(initializers, self.kernel_initializer)() if isinstance(self.kernel_initializer, str) else self.kernel_initializer
        module.weight.data = kernel_initializer(list(module.weight.shape))  # not using kernel_initializer(module.weight) for compiling with TF
        return module

    def compute_output_shape(self, input_shape):
        output_size = compute_conv_output_size(input_shape, self.kernel_size, self.strides, self.padding, dilation_rate=self.dilation_rate)
        return [None, self.filters, *output_size]

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "use_bias": self.use_bias, "groups": self.groups})
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


class Conv3D(Conv):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (4, 3, 0, 1, 2))
        return self.set_weights(weights)


class ConvTranspose(_BaseConvPool):
    def __init__(
        self, filters, kernel_size=1, strides=1, padding=0, output_padding=None, dilation_rate=1, use_bias=True, kernel_initializer="glorot_uniform", **kwargs
    ):
        super().__init__(kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, **kwargs)
        self.filters, self.use_bias, self.output_padding, self.kernel_initializer = int(filters), use_bias, output_padding, kernel_initializer
        self.output_padding = to_tuple(output_padding, num_dims=2)
        self.module_class = None
        super().__init__(**kwargs)

    def build_module(self, input_shape):
        if self.module_class is None:
            self.module_class = nn.ConvTranspose1d if len(input_shape) == 3 else (nn.ConvTranspose2d if len(input_shape) == 4 else nn.ConvTranspose3d)

        module = self.module_class(
            in_channels=input_shape[1],
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._pad,
            output_padding=0 if self.output_padding[0] is None else self.output_padding,
            dilation=self.dilation_rate,
            bias=self.use_bias,
        )
        kernel_initializer = getattr(initializers, self.kernel_initializer)() if isinstance(self.kernel_initializer, str) else self.kernel_initializer
        module.weight.data = kernel_initializer(list(module.weight.shape))
        return module

    def deconv_output_length(self, size, kernel_size, strides=1, pad=0, output_padding=None, dilation=1):
        # keras.utils.conv_utils.deconv_output_length
        if size is None:
            return None

        # Get the dilated kernel size
        kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

        # Infer length if output padding is None, else compute the exact length
        if output_padding is None:
            length = size * strides
            if pad == 0:  # "valid"
                length += max(kernel_size - strides, 0)
        else:
            length = (size - 1) * strides + kernel_size
            if padding > 0:
                length += kernel_size // 2 * 2 + output_padding
        return length

    def compute_output_shape(self, input_shape):
        input_size = input_shape[2:]
        out = [self.deconv_output_length(*args) for args in zip(input_size, self.kernel_size, self.strides, self._pad, self.output_padding, self.dilation_rate)]
        return [None, self.filters, *out]

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
        return self.set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "use_bias": self.use_bias, "output_padding": self.output_padding})
        return config


class Conv1DTranspose(ConvTranspose):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (2, 1, 0))
        return self.set_weights(weights)


class Conv2DTranspose(ConvTranspose):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
        return self.set_weights(weights)


class Conv3DTranspose(ConvTranspose):
    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (4, 3, 0, 1, 2))
        return self.set_weights(weights)


class DepthwiseConv2D(Conv):
    def __init__(self, kernel_size=1, strides=1, padding="VALID", dilation_rate=(1, 1), use_bias=True, depthwise_initializer="glorot_uniform", **kwargs):
        filters, groups = -1, -1
        super().__init__(filters, kernel_size, strides, padding, dilation_rate, use_bias, groups, depthwise_initializer, **kwargs)

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
        self.units, self.activation, self.use_bias, self.axis, self.kernel_initializer = int(units), activation, use_bias, axis, kernel_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        module = nn.Linear(in_features=input_shape[self.axis], out_features=self.units, bias=self.use_bias)
        kernel_initializer = getattr(initializers, self.kernel_initializer)() if isinstance(self.kernel_initializer, str) else self.kernel_initializer
        module.weight.data = kernel_initializer(list(module.weight.shape))  # not using kernel_initializer(module.weight) for compiling with TF
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


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer="random_uniform", mask_zero=False, input_length=None, **kwargs):
        self.input_dim, self.output_dim, self.mask_zero, self.input_length = input_dim, output_dim, mask_zero, input_length
        self.embeddings_initializer = embeddings_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.output_dim, padding_idx=None, max_norm=None)
        initializer = getattr(initializers, self.embeddings_initializer)() if isinstance(self.embeddings_initializer, str) else self.embeddings_initializer
        self.module.weight.data = initializer(list(self.module.weight.shape))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        return input_shape + [self.output_dim]

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim, "mask_zero": self.mask_zero, "input_length": self.input_length})
        return config


class GroupNormalization(Layer):
    def __init__(self, groups=32, axis=1, epsilon=0.001, center=True, gamma_initializer="ones", **kwargs):
        self.groups, self.axis, self.epsilon, self.center, self.gamma_initializer = groups, axis, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape) + self.axis if self.axis < 0 else self.axis
        module = nn.GroupNorm(num_groups=self.groups, num_channels=input_shape[self.axis], eps=self.epsilon, affine=self.center)
        gamma_initializer = getattr(initializers, self.gamma_initializer)() if isinstance(self.gamma_initializer, str) else self.gamma_initializer
        module.weight.data = gamma_initializer(list(module.weight.shape))  # not using gamma_initializer(module.weight) for compiling with TF
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


class _LayerNormGeneral(nn.Module):
    """LayerNorm supports `bias=False`, also applying on `axis=1` directly without permute.
    From LayerNormGeneral https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L311
    """

    def __init__(self, normalized_shape, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5):
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
        if self.axis == len(input_shape) - 1 and self.center:
            self.module = nn.LayerNorm(normalized_shape=input_shape[self.axis], eps=self.epsilon)
            gamma_initializer = getattr(initializers, self.gamma_initializer)() if isinstance(self.gamma_initializer, str) else self.gamma_initializer
            self.module.weight.data = gamma_initializer(list(self.module.weight.shape))
            # Default nn.LayerNorm is apllied on last dimension
            # if self.axis != len(input_shape) - 1:
            #     ndims = len(input_shape)
            #     perm = get_perm(total_axis=ndims, from_axis=self.axis, to_axis=-1)  # like axis=1 -> [0, 2, 3, 1]
            #     revert_perm = get_perm(total_axis=ndims, from_axis=-1, to_axis=self.axis)  # like axis=1 -> [0, 3, 1, 2]
            #     self.module = nn.Sequential(Permute(perm[1:]), self.module, Permute(revert_perm[1:]))
        else:
            normalized_shape = input_shape[self.axis]
            if self.axis != len(input_shape) - 1:
                normalized_shape = [normalized_shape] + [1] * (len(input_shape) - self.axis - 1)
            self.module = _LayerNormGeneral(normalized_shape=normalized_shape, normalized_dim=self.axis, eps=self.epsilon, bias=self.center)
            gamma_initializer = getattr(initializers, self.gamma_initializer)() if isinstance(self.gamma_initializer, str) else self.gamma_initializer
            self.module.weight.data = gamma_initializer(list(self.module.weight.shape))  # not using gamma_initializer(self.module.weight) for compiling with TF
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon, "center": self.center})
        return config

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights = [np.squeeze(ww) for ww in weights]
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights = [np.reshape(ss, tt.shape) for ss, tt in zip(weights, self.weights)]
        return self.set_weights(weights)


class PReLU(Layer):
    def __init__(self, alpha_initializer="zeros", alpha_regularizer=None, alpha_constraint=None, shared_axes=None, **kwargs):
        self.alpha_initializer, self.shared_axes = alpha_initializer, shared_axes
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.PReLU(num_parameters=input_shape[1], init=0.25)
        alpha_initializer = getattr(initializers, self.alpha_initializer)() if isinstance(self.alpha_initializer, str) else self.alpha_initializer
        self.module.weight.data = alpha_initializer(list(self.module.weight.shape))  # not using alpha_initializer(self.module.weight) for compiling with TF
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


class SeparableConv2D(Conv):
    def __init__(self, filters, kernel_size=1, strides=1, padding="VALID", dilation_rate=1, use_bias=True, pointwise_initializer="glorot_uniform", **kwargs):
        groups = 1
        super().__init__(filters, kernel_size, strides, padding, dilation_rate, use_bias, groups, pointwise_initializer, **kwargs)

    def build_module(self, input_shape):
        depthwise = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=input_shape[1],
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._pad,
            dilation=self.dilation_rate,
            groups=input_shape[1],
            bias=False,
        )
        pointwise = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=self.filters,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=self.use_bias,
        )

        kernel_initializer = getattr(initializers, self.kernel_initializer)() if isinstance(self.kernel_initializer, str) else self.kernel_initializer
        depthwise.weight.data = kernel_initializer(list(depthwise.weight.shape))
        pointwise.weight.data = kernel_initializer(list(pointwise.weight.shape))
        return nn.Sequential(depthwise, pointwise)

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
        weights[1] = np.transpose(weights[1], (2, 3, 1, 0))
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))
        return self.set_weights(weights)


""" Layers with no weights """


class _DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    from drop_path https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/drop.py#L137
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
        super().__init__()

    def forward(self, inputs):
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = inputs.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return inputs * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Dropout(Layer):
    def __init__(self, rate, noise_shape=None, **kwargs):
        self.rate, self.noise_shape = rate, noise_shape
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.noise_shape is None:
            dims, rate = len(input_shape), self.rate
            # self.module = torch.nn.Dropout1d(p=rate) if dims == 3 else (torch.nn.Dropout2d(p=rate) if dims == 4 else torch.nn.Dropout3d(p=rate))
            self.module = torch.nn.Dropout(p=rate)
        else:
            self.module = _DropPath(drop_prob=self.rate)
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


class _Pooling2D(_BaseConvPool):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", reduce="mean", **kwargs):
        super().__init__(kernel_size=pool_size, strides=strides, padding=padding, dilation_rate=(1, 1), **kwargs)
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else [pool_size, pool_size]
        self.reduce = reduce

    def build_module(self, input_shape):
        if self.reduce.lower() == "max":
            module = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.strides, padding=self._pad)
        else:
            module = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.strides, padding=self._pad, count_include_pad=False)
        return module

    def compute_output_shape(self, input_shape):
        output_size = compute_conv_output_size(input_shape, self.pool_size, self.strides, self.padding, dilation_rate=1)
        return [None, input_shape[1], *output_size]

    def get_config(self):
        config = super().get_config()
        config.pop("kernel_size")  # From super
        config.pop("dilation_rate")  # From super
        config.update({"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding})  # Not saving reduce
        return config


class AvgPool2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, reduce=kwargs.pop("reduce", "mean"), **kwargs)


class MaxPool2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, reduce=kwargs.pop("reduce", "max"), **kwargs)


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


class LeakyReLU(Layer):
    def __init__(self, alpha=0.3, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.LeakyReLU(negative_slope=self.alpha)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
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

    def extra_repr(self):
        return f"dims={self.dims}"


class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        self.target_shape = [-1 if ii is None else ii for ii in target_shape]
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_unknown_dim = sum([ii == -1 for ii in self.target_shape])
        assert num_unknown_dim < 2, "At most one unknown dimension in output_shape: {}".format(self.target_shape)

        if any([ii is None or ii == -1 for ii in input_shape[1:]]):  # Dynamic input_shape
            if num_unknown_dim > 0:
                knwon_target_dim = -1 * np.prod(self.target_shape)
                self.module = partial(
                    lambda inputs: inputs.contiguous().view(
                        [inputs.shape[0]] + [inputs.shape[1:].numel() // knwon_target_dim if ii == -1 else ii for ii in self.target_shape]
                    )
                )
            else:
                self.module = partial(lambda inputs: inputs.contiguous().view([inputs.shape[0], *self.target_shape]))
        else:
            total_size = np.prod(input_shape[1:])
            if num_unknown_dim > 0:
                unknown_dim = total_size // (-1 * np.prod(self.target_shape))
                self.target_shape = [unknown_dim if ii == -1 else ii for ii in self.target_shape]
            assert total_size == np.prod(self.target_shape), "Total size of new array must be unchanged, {} -> {}".format(input_shape, self.target_shape)
            self.module = partial(lambda inputs: inputs.contiguous().view([-1, *self.target_shape]))

        # self.module = partial(torch.reshape, shape=[-1, *self.target_shape])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape[0]] + [None if ii == -1 else ii for ii in self.target_shape]

    def get_config(self):
        config = super().get_config()
        config.update({"target_shape": self.target_shape})
        return config

    def extra_repr(self):
        return f"target_shape={self.target_shape}"


class _ResizeDynamic(Layer):
    """
    Resize supporting dynamic size.
    >>> import torch
    >>> from keras_cv_attention_models.pytorch_backend import layers
    >>> inputs = layers.Input([3, 16, 16])
    >>> size = layers.Shape(inputs)[2:]
    >>> aa = layers._ResizeDynamic()
    >>> print(f"{aa([torch.ones([1, 3, 24, 24]), np.array(size)]).shape = }")
    >>> # aa(torch.ones([1, 3, 24, 24])).shape = torch.Size([1, 3, 16, 16])
    >>> inputs.set_shape([None, 1, 32, 32])
    >>> print(f"{aa([torch.ones([1, 3, 24, 24]), np.array(size)]).shape = }")
    >>> # aa(torch.ones([1, 3, 24, 24])).shape = torch.Size([1, 3, 32, 32])
    """

    def __init__(self, method="bilinear", preserve_aspect_ratio=False, antialias=False, **kwargs):
        self.method, self.preserve_aspect_ratio, self.antialias = method, preserve_aspect_ratio, antialias
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2, "Input should be in format [inputs, size]"
        input_shape, size_shape = input_shape[0], input_shape[1]

        size_len = size_shape[0] if isinstance(size_shape, (list, tuple)) else 1
        assert len(input_shape) - 2 == size_len, f"Provided input_shape={input_shape} length should be larger than size_shape={size_shape} by 2"
        if self.antialias:
            assert len(input_shape) == 4, "Anti-alias option is only supported for 2D resize"
            assert self.method in ["bilinear", "bicubic"], "Anti-alias option is only supported for bilinear and bicubic modes"

        self.module = lambda inputs: torch.functional.F.interpolate(inputs[0], size=list(inputs[1]), mode=self.method, antialias=self.antialias)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # input_shape = list(input_shape[0])
        return list(input_shape[0][:2]) + [None] * (len(input_shape[0]) - 2)  # Dynamic shape

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method, "preserve_aspect_ratio": self.preserve_aspect_ratio, "antialias": self.antialias})
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


class UpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, interpolation="nearest", **kwargs):
        self.data_format, self.interpolation = data_format, interpolation
        self.size = tuple(size if isinstance(size, (list, tuple)) else [size, size])
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.Upsample(scale_factor=self.size, mode=self.interpolation)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size, "data_format": self.data_format, "interpolation": self.interpolation})
        return config

    def compute_output_shape(self, input_shape):
        return [None, input_shape[1], input_shape[2] * self.size[0], input_shape[3] * self.size[1]]


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
        hh = None if input_shape[2] is None else (input_shape[2] + self.padding[0] * 2)
        ww = None if input_shape[3] is None else (input_shape[3] + self.padding[1] * 2)
        return [input_shape[0], input_shape[1], hh, ww]

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config
