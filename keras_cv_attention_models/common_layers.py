import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, initializers, image_data_format

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
TF_BATCH_NORM_EPSILON = 0.001
LAYER_NORM_EPSILON = 1e-5


""" Wrapper for default parameters """


@backend.register_keras_serializable(package="kecamCommon")
def hard_swish(inputs):
    """`out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244"""
    return inputs * functional.relu6(inputs + 3) / 6


@backend.register_keras_serializable(package="kecamCommon")
def hard_sigmoid_torch(inputs):
    """https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    toch.nn.Hardsigmoid: 0 if x <= −3 else (1 if x >= 3 else x / 6 + 1/2)
    keras.activations.hard_sigmoid: 0 if x <= −2.5 else (1 if x >= 2.5 else x / 5 + 1/2) -> tf.clip_by_value(inputs / 5 + 0.5, 0, 1)
    """
    return functional.clip_by_value(inputs / 6 + 0.5, 0, 1)


@backend.register_keras_serializable(package="kecamCommon")
def mish(inputs):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    Copied from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/mish.py
    """
    return inputs * functional.tanh(functional.softplus(inputs))


@backend.register_keras_serializable(package="kecamCommon")
def phish(inputs):
    """Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    Paper: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
    return inputs * functional.tanh(functional.gelu(inputs))


def gelu_quick(inputs):
    """https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90-L98
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return inputs * functional.sigmoid(1.702 * inputs)


def gelu_linear(inputs):
    """
    >>> from keras_cv_attention_models.common_layers import gelu_linear
    >>> xx = np.arange(-4, 4, 0.01)
    >>> plt.plot(xx, tf.nn.gelu(xx), label='gelu')
    >>> plt.plot(xx, tf.nn.gelu(xx, approximate=True), label='gelu, approximate')
    >>> plt.plot(xx, gelu_linear(xx), label='gelu_linear')
    >>> plt.legend()
    >>> plt.grid(True)
    """
    inputs_abs = functional.abs(inputs)
    inputs_sign = functional.sign(inputs)

    erf = inputs_abs * -0.7071
    erf = functional.relu(erf + 1.769)
    erf = erf**2 * -0.1444 + 0.5
    return inputs * (erf * inputs_sign + 0.5)


def activation_by_name(inputs, activation="relu", name=None):
    """Typical Activation layer added hard_swish and prelu."""
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    activation_lower = activation.lower()
    if activation_lower == "hard_swish":
        return layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    if activation_lower == "leaky_relu":
        return layers.LeakyReLU(name=layer_name)(inputs)
    elif activation_lower == "mish":
        return layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation_lower == "phish":
        return layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation_lower == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if backend.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return layers.PReLU(shared_axes=shared_axes, alpha_initializer=initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation_lower.startswith("gelu/app"):
        # gelu/approximate
        return functional.gelu(inputs, approximate=True)
    elif activation_lower.startswith("gelu/linear"):
        return gelu_linear(inputs)
    elif activation_lower.startswith("gelu/quick"):
        return gelu_quick(inputs)
    elif activation_lower.startswith("leaky_relu/"):
        # leaky_relu with alpha parameter
        alpha = float(activation_lower.split("/")[-1])
        return layers.LeakyReLU(alpha=alpha, name=layer_name)(inputs)
    elif activation_lower == ("hard_sigmoid_torch"):
        return layers.Activation(activation=hard_sigmoid_torch, name=layer_name)(inputs)
    elif activation_lower == ("squaredrelu") or activation_lower == ("squared_relu"):
        return functional.pow(functional.relu(inputs), 2)  # Squared ReLU: https://arxiv.org/abs/2109.08668
    elif activation_lower == ("starrelu") or activation_lower == ("star_relu"):
        from keras_cv_attention_models.nfnets.nfnets import ZeroInitGain

        # StarReLU: s * relu(x) ** 2 + b
        return ZeroInitGain(use_bias=True, weight_init_value=1.0, name=layer_name)(functional.pow(functional.relu(inputs), 2))
    else:
        return layers.Activation(activation=activation, name=layer_name)(inputs)


@backend.register_keras_serializable(package="kecamCommon")
class EvoNormalization(layers.Layer):
    def __init__(self, nonlinearity=True, num_groups=-1, zero_gamma=False, momentum=0.99, epsilon=0.001, data_format="auto", **kwargs):
        # [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py)
        # EVONORM_B0: nonlinearity=True, num_groups=-1
        # EVONORM_S0: nonlinearity=True, num_groups > 0
        # EVONORM_B0 / EVONORM_S0 linearity: nonlinearity=False, num_groups=-1
        # EVONORM_S0A linearity: nonlinearity=False, num_groups > 0
        super().__init__(**kwargs)
        self.data_format, self.nonlinearity, self.zero_gamma, self.num_groups = data_format, nonlinearity, zero_gamma, num_groups
        self.momentum, self.epsilon = momentum, epsilon
        self.is_channels_first = (
            True if data_format == "channels_first" or (data_format == "auto" and backend.image_data_format() == "channels_first") else False
        )

    def build(self, input_shape):
        all_axes = list(range(len(input_shape)))
        param_shape = [1] * len(input_shape)
        if self.is_channels_first:
            param_shape[1] = input_shape[1]
            self.reduction_axes = all_axes[:1] + all_axes[2:]
        else:
            param_shape[-1] = input_shape[-1]
            self.reduction_axes = all_axes[:-1]

        self.gamma = self.add_weight(name="gamma", shape=param_shape, initializer="zeros" if self.zero_gamma else "ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=param_shape, initializer="zeros", trainable=True)
        if self.num_groups <= 0:  # EVONORM_B0
            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                initializer="ones",
                # synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                # aggregation=tf.VariableAggregation.MEAN,
            )
        if self.nonlinearity:
            self.vv = self.add_weight(name="vv", shape=param_shape, initializer="ones", trainable=True)

        if self.num_groups > 0:  # EVONORM_S0
            channels_dim = input_shape[1] if self.is_channels_first else input_shape[-1]
            num_groups = int(self.num_groups)
            while num_groups > 1:
                if channels_dim % num_groups == 0:
                    break
                num_groups -= 1
            self.__num_groups__ = num_groups
            self.groups_dim = channels_dim // self.__num_groups__

            if self.is_channels_first:
                self.group_shape = [-1, self.__num_groups__, self.groups_dim, *input_shape[2:]]
                self.group_reduction_axes = list(range(2, len(self.group_shape)))  # [2, 3, 4]
                self.group_axes = 2
                self.var_shape = [-1, *param_shape[1:]]
            else:
                self.group_shape = [-1, *input_shape[1:-1], self.__num_groups__, self.groups_dim]
                self.group_reduction_axes = list(range(1, len(self.group_shape) - 2)) + [len(self.group_shape) - 1]  # [1, 2, 4]
                self.group_axes = -1
                self.var_shape = [-1, *param_shape[1:]]

    def __group_std__(self, inputs):
        # _group_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L171
        grouped = functional.reshape(inputs, self.group_shape)
        _, var = functional.moments(grouped, self.group_reduction_axes, keepdims=True)
        std = functional.sqrt(var + self.epsilon)
        std = functional.repeat(std, self.groups_dim, axis=self.group_axes)
        return functional.reshape(std, self.var_shape)

    def __batch_std__(self, inputs, training=None):
        # _batch_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L120
        def _call_train_():
            _, var = functional.moments(inputs, self.reduction_axes, keepdims=True)
            # update_op = tf.assign_sub(moving_variance, (moving_variance - variance) * (1 - decay))
            delta = (self.moving_variance - var) * (1 - self.momentum)
            self.moving_variance.assign_sub(delta)
            return var

        def _call_test_():
            return self.moving_variance

        var = backend.in_train_phase(_call_train_, _call_test_, training=training)
        return functional.sqrt(var + self.epsilon)

    def __instance_std__(self, inputs):
        # _instance_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L111
        # axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        _, var = functional.moments(inputs, self.reduction_axes[1:], keepdims=True)
        return functional.sqrt(var + self.epsilon)

    def call(self, inputs, training=None, **kwargs):
        if self.nonlinearity and self.num_groups > 0:  # EVONORM_S0
            den = self.__group_std__(inputs)
            inputs = inputs * functional.sigmoid(self.vv * inputs) / den
        elif self.num_groups > 0:  # EVONORM_S0a
            # EvoNorm2dS0a https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/evo_norm.py#L239
            den = self.__group_std__(inputs)
            inputs = inputs / den
        elif self.nonlinearity:  # EVONORM_B0
            left = self.__batch_std__(inputs, training)
            right = self.vv * inputs + self.__instance_std__(inputs)
            inputs = inputs / functional.maximum(left, right)
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nonlinearity": self.nonlinearity,
                "zero_gamma": self.zero_gamma,
                "num_groups": self.num_groups,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "data_format": self.data_format,
            }
        )
        return config


def batchnorm_with_activation(
    inputs, activation=None, zero_gamma=False, epsilon=1e-5, momentum=0.9, axis="auto", act_first=False, use_evo_norm=False, evo_norm_group_size=-1, name=None
):
    """Performs a batch normalization followed by an activation."""
    if use_evo_norm:
        nonlinearity = False if activation is None else True
        num_groups = inputs.shape[-1] // evo_norm_group_size  # Currently using gorup_size as parameter only
        return EvoNormalization(nonlinearity, num_groups=num_groups, zero_gamma=zero_gamma, epsilon=epsilon, momentum=momentum, name=name + "evo_norm")(inputs)

    bn_axis = (-1 if backend.image_data_format() == "channels_last" else 1) if axis == "auto" else axis
    gamma_initializer = initializers.zeros() if zero_gamma else initializers.ones()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=momentum,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def layer_norm(inputs, zero_gamma=False, epsilon=LAYER_NORM_EPSILON, center=True, axis="auto", name=None):
    """Typical LayerNormalization with epsilon=1e-5"""
    norm_axis = (-1 if backend.image_data_format() == "channels_last" else 1) if axis == "auto" else axis
    gamma_init = initializers.zeros() if zero_gamma else initializers.ones()
    return layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, gamma_initializer=gamma_init, center=center, name=name and name + "ln")(inputs)


def group_norm(inputs, groups=32, epsilon=BATCH_NORM_EPSILON, axis="auto", name=None):
    """Typical GroupNormalization with epsilon=1e-5"""
    if hasattr(layers, "GroupNormalization"):
        GroupNormalization = layers.GroupNormalization  # GroupNormalization is added after TF 2.11.0
    else:
        from tensorflow_addons.layers import GroupNormalization

    norm_axis = (-1 if backend.image_data_format() == "channels_last" else 1) if axis == "auto" else axis
    return GroupNormalization(groups=groups, axis=norm_axis, epsilon=epsilon, name=name and name + "group_norm")(inputs)


def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="valid", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """Typical Conv2D with `use_bias` default as `False` and fixed padding,
    and torch initializer `uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)`
    """
    kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
    if isinstance(padding, str):
        padding = padding.lower()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2) if use_torch_padding and padding == "same" else (0, 0)
    else:  # int or list or tuple with specific value
        pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
        padding = "same" if max(pad) > 0 else "valid"

    if use_torch_padding and not backend.is_torch_backend and padding == "same":
        inputs = layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs) if max(pad) > 0 else inputs
        padding = "valid"

    kernel_initializer = kwargs.get("kernel_initializer", None)
    if kernel_initializer is None and not backend.is_torch_backend:
        fan_in = 1 / (float(inputs.shape[-1] * kernel_size[0] * kernel_size[1]) ** 0.5)
        kernel_initializer = initializers.RandomUniform(-fan_in, fan_in)

    groups = max(1, groups)
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="valid" if padding == "valid" else (pad if use_torch_padding else "same"),
        use_bias=use_bias,
        groups=groups,
        kernel_initializer=kernel_initializer,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="valid", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding
    and torch initializer `uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)`
    """
    kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
    if isinstance(padding, str):
        padding = padding.lower()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2) if use_torch_padding and padding == "same" else (0, 0)
    else:  # int or list or tuple with specific value
        pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
        padding = "same" if max(pad) > 0 else "valid"

    if use_torch_padding and not backend.is_torch_backend and padding == "same":
        inputs = layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs) if max(pad) > 0 else inputs
        padding = "valid"

    depthwise_initializer = kwargs.get("depthwise_initializer", None)
    if depthwise_initializer is None and not backend.is_torch_backend:
        fan_in = 1 / (float(inputs.shape[-1] * kernel_size[0] * kernel_size[1]) ** 0.5)
        depthwise_initializer = initializers.RandomUniform(-fan_in, fan_in)

    return layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding="valid" if padding == "valid" else (pad if use_torch_padding else "same"),
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)


def dense_no_bias(inputs, units, use_bias=False, name=None, **kwargs):
    """Typical Dense with `use_bias` default as `False`, and Torch Linear initializer `uniform(-1/sqrt(in_features), 1/sqrt(in_features))`"""
    kernel_initializer = kwargs.get("kernel_initializer", None)
    if kernel_initializer is None and not backend.is_torch_backend:
        fan_in = 1 / (float(inputs.shape[-1]) ** 0.5)
        kernel_initializer = initializers.RandomUniform(-fan_in, fan_in)
    return layers.Dense(units, kernel_initializer=kernel_initializer, use_bias=use_bias, name=name, **kwargs)(inputs)


""" Blocks """


def output_block(inputs, filters=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True, act_first=False):
    nn = inputs
    if filters > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, filters, 1, strides=1, use_bias=act_first, use_torch_padding=is_torch_mode, name="features_")  # Also use_bias for act_first
        nn = batchnorm_with_activation(nn, activation=activation, act_first=act_first, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn) if len(nn.shape) == 4 else nn
        if drop_rate > 0:
            nn = layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn


def global_context_module(inputs, use_attn=True, ratio=0.25, divisor=1, activation="relu", use_bias=True, name=None):
    """Global Context Attention Block, arxiv: https://arxiv.org/pdf/1904.11492.pdf"""
    is_channels_last = image_data_format() == "channels_last"
    filters = inputs.shape[-1 if is_channels_last else 1]
    height_axis, width_axis = (1, 2) if is_channels_last else (2, 3)
    height, width = inputs.shape[height_axis], inputs.shape[width_axis]

    # activation could be ("relu", "hard_sigmoid")
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    reduction = make_divisible(filters * ratio, divisor, limit_round_down=0.0)

    if use_attn:
        attn = layers.Conv2D(1, kernel_size=1, use_bias=use_bias, name=name and name + "attn_conv")(inputs)
        attn = functional.reshape(attn, [-1, 1, height * width])  # [batch, height, width, 1] or [batch, 1, height, width] -> [batch, 1, height * width]
        attn = functional.softmax(attn, axis=-1)
        context = inputs if is_channels_last else functional.transpose(inputs, [0, 2, 3, 1])
        context = functional.reshape(context, [-1, height * width, filters])
        context = attn @ context  # [batch, 1, filters]
        context = functional.reshape(context, [-1, 1, 1, filters]) if is_channels_last else functional.reshape(context, [-1, filters, 1, 1])
    else:
        context = functional.reduce_mean(inputs, [height_axis, width_axis], keepdims=True)

    mlp = layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_1_conv")(context)
    mlp = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(mlp)
    mlp = activation_by_name(mlp, activation=hidden_activation, name=name)
    mlp = layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_2_conv")(mlp)
    mlp = activation_by_name(mlp, activation=output_activation, name=name)
    return layers.Multiply(name=name and name + "out")([inputs, mlp])


def se_module(inputs, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, use_conv=True, name=None):
    """Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf"""
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]

    # activation could be ("relu", "hard_sigmoid") for mobilenetv3
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    filters = inputs.shape[channel_axis]
    reduction = make_divisible(filters * se_ratio, divisor, limit_round_down=limit_round_down)
    # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
    se = functional.reduce_mean(inputs, [h_axis, w_axis], keepdims=True if use_conv else False)
    if use_conv:
        se = layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, name=name and name + "1_conv")(se)
    else:
        se = layers.Dense(reduction, use_bias=use_bias, name=name and name + "1_dense")(se)
    se = activation_by_name(se, activation=hidden_activation, name=name)
    if use_conv:
        se = layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name and name + "2_conv")(se)
    else:
        se = layers.Dense(filters, use_bias=use_bias, name=name and name + "2_dense")(se)
    se = activation_by_name(se, activation=output_activation, name=name)
    se = se if use_conv else functional.reshape(se, [-1, 1, 1, filters] if image_data_format() == "channels_last" else [-1, filters, 1, 1])
    return layers.Multiply(name=name and name + "out")([inputs, se])


def eca_module(inputs, gamma=2.0, beta=1.0, name=None, **kwargs):
    """Efficient Channel Attention block, arxiv: https://arxiv.org/pdf/1910.03151.pdf"""
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]

    filters = inputs.shape[channel_axis]
    beta, gamma = float(beta), float(gamma)
    tt = int((np.log(float(filters)) / np.log(2.0) + beta) / gamma)
    kernel_size = max(tt if tt % 2 else tt + 1, 3)
    pad = kernel_size // 2

    nn = functional.reduce_mean(inputs, [h_axis, w_axis], keepdims=False)
    nn = functional.pad(nn, [[0, 0], [pad, pad]])
    nn = functional.expand_dims(nn, channel_axis)

    nn = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="valid", use_bias=False, name=name and name + "conv1d")(nn)
    nn = functional.squeeze(nn, axis=channel_axis)
    nn = activation_by_name(nn, activation="sigmoid", name=name)
    nn = nn[:, None, None] if image_data_format() == "channels_last" else nn[:, :, None, None]
    # print(f"{inputs.shape = }, {nn.shape = }")
    return layers.Multiply(name=name and name + "out")([inputs, nn])


def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """split drop connect rate in range `(start, end)` according to `num_blocks`"""
    # drop_connect_rates = functional.split(functional.linspace(start, end, sum(num_blocks)), num_blocks)
    cum_split = [sum(num_blocks[: id + 1]) for id, _ in enumerate(num_blocks[:-1])]
    drop_connect_rates = np.split(np.linspace(start, end, sum(num_blocks)), cum_split)
    return [ii.tolist() for ii in drop_connect_rates]


def drop_block(inputs, drop_rate=0, name=None):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs


def addaptive_pooling_2d(inputs, output_size, reduce="mean", data_format="auto", name=None):
    """Auto set `pool_size` and `strides` for `MaxPool2D` or `AvgPool2D` fitting `output_size`.
    (in_height - (pool_size - strides)) / strides == out_height
    condition: pool_size >= strides, pool_size != 0, strides != 0
    strides being as large as possible: strides == in_height // out_height
    ==> pool_size = in_height - (out_height - 1) * strides, not in_height % strides, in case in_height == strides  will be 0
    """
    data_format = image_data_format() if data_format == "auto" else data_format
    height, width = inputs.shape[1:-1] if image_data_format() == "channels_last" else inputs.shape[2:]
    h_bins, w_bins = output_size[:2] if isinstance(output_size, (list, tuple)) else (output_size, output_size)
    reduce_function = layers.MaxPool2D if reduce.lower() == "max" else layers.AvgPool2D

    h_strides, w_strides = height // h_bins, width // w_bins
    h_pool_size, w_pool_size = height - (h_bins - 1) * h_strides, width - (w_bins - 1) * w_strides
    # print(f"{inputs.shape = }, {h_pool_size = }, {w_pool_size = }, {h_strides = }, {w_strides = }")
    return reduce_function(pool_size=(h_pool_size, w_pool_size), strides=(h_strides, w_strides), name=name and name + "pool")(inputs)


""" Other layers / functions """


@backend.register_keras_serializable(package="kecamCommon")
class AntiAliasDownsampleInitializer(initializers.Initializer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, weight_shape, dtype="float32", **kwargs):
        kernel_size, channel = (weight_shape[0], weight_shape[2]) if backend.image_data_format() == "channels_last" else (weight_shape[2], weight_shape[0])
        ww = np.array(np.poly1d((0.5, 0.5)) ** (kernel_size - 1)).astype("float32")
        ww = np.expand_dims(ww, 0) * np.expand_dims(ww, 1)
        if backend.image_data_format() == "channels_last":
            ww = np.repeat(ww[:, :, None, None], channel, axis=-2)
        else:
            ww = np.repeat(ww[None, None, :, :], channel, axis=0)
        return functional.convert_to_tensor(ww, dtype=dtype)


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="same", trainable=False, name=None):
    """DepthwiseConv2D performing anti-aliasing downsample, arxiv: https://arxiv.org/pdf/1904.11486.pdf"""
    return layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        trainable=trainable,
        depthwise_initializer=AntiAliasDownsampleInitializer(),
        name=name and name + "anti_alias_down",
    )(inputs)


def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
    """Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < limit_round_down * vv:
        new_v += divisor
    return new_v


@backend.register_keras_serializable(package="kecamCommon")
class UnfoldFiltersInitializer(initializers.Initializer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, weight_shape, dtype="float32", **kwargs):
        kernel_size = weight_shape[0]
        kernel_out = kernel_size * kernel_size
        ww = np.reshape(np.eye(kernel_out, dtype="float32"), [kernel_size, kernel_size, 1, kernel_out])
        if len(weight_shape) == 5:  # Conv3D or Conv3DTranspose
            ww = np.expand_dims(ww, 2)
        return functional.convert_to_tensor(ww)


def fold_by_conv2d_transpose(patches, output_shape=None, kernel_size=3, strides=2, dilation_rate=1, padding="same", compressed="auto", name=None):
    paded = kernel_size // 2 if padding else 0
    if compressed == "auto":
        compressed = True if len(patches.shape) == 4 else False

    if compressed:
        _, hh, ww, cc = patches.shape
        channel = cc // kernel_size // kernel_size
        conv_rr = functional.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    else:
        _, hh, ww, _, _, channel = patches.shape
        # conv_rr = patches
        conv_rr = functional.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    conv_rr = functional.transpose(conv_rr, [0, 3, 1, 2])  # [batch, channnel, hh * ww, kernel * kernel]
    conv_rr = functional.reshape(conv_rr, [-1, hh, ww, kernel_size * kernel_size])

    convtrans_rr = layers.Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="valid",
        output_padding=paded,
        use_bias=False,
        trainable=False,
        kernel_initializer=UnfoldFiltersInitializer(),
        name=name and name + "fold_convtrans",
    )(conv_rr)

    out = functional.reshape(functional.squeeze(convtrans_rr, axis=-1), [-1, channel, convtrans_rr.shape[1], convtrans_rr.shape[2]])
    out = functional.transpose(out, [0, 2, 3, 1])
    if output_shape is None:
        output_shape = [-paded, -paded]
    else:
        output_shape = [output_shape[0] + paded, output_shape[1] + paded]
    out = out[:, paded : output_shape[0], paded : output_shape[1]]
    return out


@backend.register_keras_serializable(package="kecamCommon")
class CompatibleExtractPatches(layers.Layer):
    def __init__(self, sizes=3, strides=2, rates=1, padding="same", compressed=True, force_conv=False, **kwargs):
        super().__init__(**kwargs)
        self.sizes, self.strides, self.rates, self.padding = sizes, strides, rates, padding
        self.compressed, self.force_conv = compressed, force_conv

        self.kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
        self.strides = strides[1] if isinstance(strides, (list, tuple)) else strides
        # dilation_rate can be 2 different values, used in DiNAT
        self.dilation_rate = (rates if len(rates) == 2 else rates[1:3]) if isinstance(rates, (list, tuple)) else (rates, rates)
        self.filters = self.kernel_size * self.kernel_size

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            if len(tf.config.experimental.list_logical_devices("TPU")) != 0 or self.force_conv:
                self.use_conv = True
            else:
                self.use_conv = False
        else:
            self.use_conv = force_conv

    def build(self, input_shape):
        _, self.height, self.width, self.channel = input_shape
        if self.padding.lower() == "same":
            pad_value = self.kernel_size // 2
            self.pad_value_list = [[0, 0], [pad_value, pad_value], [pad_value, pad_value], [0, 0]]
            self.height, self.width = self.height + pad_value * 2, self.width + pad_value * 2
            self.pad_value = pad_value
        else:
            self.pad_value = 0

        if self.use_conv:
            self.conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding="valid",
                use_bias=False,
                trainable=False,
                kernel_initializer=UnfoldFiltersInitializer(),
                name=self.name and self.name + "unfold_conv",
            )
            self.conv.build([None, *input_shape[1:-1], 1])
        else:
            self._sizes_ = [1, self.kernel_size, self.kernel_size, 1]
            self._strides_ = [1, self.strides, self.strides, 1]
            self._rates_ = [1, *self.dilation_rate, 1]
        # output_size = backend.compute_conv_output_size([self.height, self.width], self.kernel_size, self.strides, self.padding, self.dilation_rate)
        # self.output_height, self.output_width = output_size
        super().build(input_shape)

    def call(self, inputs):
        if self.pad_value > 0:
            inputs = functional.pad(inputs, self.pad_value_list)

        if self.use_conv:
            merge_channel = functional.transpose(inputs, [0, 3, 1, 2])
            merge_channel = functional.reshape(merge_channel, [-1, self.height, self.width, 1])
            conv_rr = self.conv(merge_channel)

            # TFLite not supporting `tf.transpose` with len(perm) > 4...
            out = functional.reshape(conv_rr, [-1, self.channel, conv_rr.shape[1] * conv_rr.shape[2], self.filters])
            out = functional.transpose(out, [0, 2, 3, 1])  # [batch, hh * ww, kernel * kernel, channnel]
            if self.compressed:
                out = functional.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.filters * self.channel])
            else:
                out = functional.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.kernel_size, self.kernel_size, self.channel])
        else:
            out = functional.extract_patches(inputs, self._sizes_, self._strides_, self._rates_, "VALID")  # must be upper word VALID/SAME
            if not self.compressed:
                # [batch, hh, ww, kernel, kernel, channnel]
                out = functional.reshape(out, [-1, out.shape[1], out.shape[2], self.kernel_size, self.kernel_size, self.channel])
        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "sizes": self.sizes,
                "strides": self.strides,
                "rates": self.rates,
                "padding": self.padding,
                "compressed": self.compressed,
                "force_conv": self.force_conv,
            }
        )
        return base_config


""" Preprocess input and decode predictions """


def init_mean_std_by_rescale_mode(rescale_mode, convert_to_image_data_format=True):
    if isinstance(rescale_mode, (list, tuple)):  # Specific mean and std
        mean, std = rescale_mode
    elif rescale_mode == "torch":
        mean = np.array([0.485, 0.456, 0.406]).astype("float32") * 255.0
        std = np.array([0.229, 0.224, 0.225]).astype("float32") * 255.0
        if backend.image_data_format() != "channels_last" and convert_to_image_data_format:
            mean, std = mean[:, None, None], std[:, None, None]
    elif rescale_mode == "tf":  # [0, 255] -> [-1, 1]
        mean, std = 127.5, 127.5
        # mean, std = 127.5, 128.0
    elif rescale_mode == "tf128":  # [0, 255] -> [-1, 1]
        mean, std = 128.0, 128.0
    elif rescale_mode == "raw01":
        mean, std = 0, 255.0  # [0, 255] -> [0, 1]
    elif rescale_mode == "clip":  # value from openai/CLIP
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).astype("float32") * 255.0
        std = np.array([0.26862954, 0.26130258, 0.27577711]).astype("float32") * 255.0
        if backend.image_data_format() != "channels_last" and convert_to_image_data_format:
            mean, std = mean[:, None, None], std[:, None, None]
    else:
        mean, std = 0, 1  # raw inputs [0, 255]
    return mean, std


class PreprocessInput:
    """`rescale_mode` `torch` means `(image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`, `tf` means `(image - 0.5) / 0.5`"""

    def __init__(self, input_shape=(224, 224, 3), rescale_mode="torch"):
        self.set_input_shape(input_shape)
        self.set_rescale_mode(rescale_mode)

    def set_input_shape(self, input_shape):
        input_shape = input_shape[1:] if len(input_shape) == 4 else input_shape
        if None in input_shape:
            self.input_shape = (None, None)  # Dynamic input_shape
        elif len(input_shape) == 2:
            self.input_shape = input_shape
        else:
            channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1])  # Assume the smallest value is the channel dimension
            self.input_shape = [dim for axis, dim in enumerate(input_shape) if axis != channel_axis]

    def set_rescale_mode(self, rescale_mode):
        self.mean, self.std = init_mean_std_by_rescale_mode(rescale_mode)
        self.rescale_mode = rescale_mode

    def __call__(self, image, resize_method="bilinear", resize_antialias=False, input_shape=None):
        if input_shape is not None:
            self.set_input_shape(input_shape)
        images = np.array([image] if len(np.shape(image)) == 3 else image).astype("float32")
        images = (images * 255) if images.max() < 2 else images

        images = images if backend.image_data_format() == "channels_last" else images.transpose([0, 3, 1, 2])
        images = functional.convert_to_tensor(images)
        images = functional.resize(images, self.input_shape, method=resize_method, antialias=resize_antialias)
        images = (images - self.mean) / self.std
        return images


def add_pre_post_process(model, rescale_mode="tf", input_shape=None, post_process=None, features=None):
    from keras_cv_attention_models.imagenet.eval_func import decode_predictions

    input_shape = model.input_shape[1:] if input_shape is None else input_shape
    model.preprocess_input = PreprocessInput(input_shape, rescale_mode=rescale_mode)
    model.decode_predictions = decode_predictions if post_process is None else post_process
    model.rescale_mode = rescale_mode

    if features is not None:
        model.extract_features = lambda: features
