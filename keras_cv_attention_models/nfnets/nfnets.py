import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import activation_by_name, drop_block, se_module, make_divisible

CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
NON_LINEAR_GAMMA = dict(
    identity = 1.0,
    celu = 1.270926833152771,
    elu = 1.2716004848480225,
    gelu = 1.7015043497085571,
    leaky_relu = 1.70590341091156,
    log_sigmoid = 1.9193484783172607,
    log_softmax = 1.0002083778381348,
    relu = 1.7139588594436646,
    relu6 = 1.7131484746932983,
    selu = 1.0008515119552612,
    sigmoid = 4.803835391998291,
    swish = 1.7881293296813965, # silu
    softsign = 2.338853120803833,
    softplus = 1.9203323125839233,
    tanh = 1.5939117670059204,
)

class ScaledStandardizedConv2D(tf.keras.layers.Conv2D):
    """
    Copied from https://github.com/google-research/big_transfer/blob/master/bit_tf2/models.py, Author: Lucas Beyer
    Modified reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """
    def __init__(self, eps=1e-4, *args, **kwargs):
        super(ScaledStandardizedConv2D, self).__init__(*args, **kwargs)
        self.eps = eps
        self.__eps__ = tf.cast(eps, self.dtype)

    def build(self, input_shape):
        super(ScaledStandardizedConv2D, self).build(input_shape)
        # Wrap a standardization around the conv OP.
        default_conv_op = self._convolution_op
        self.gain = self.add_weight(name='gain', shape=(self.filters,), initializer="ones", trainable=True, dtype=self.dtype)
        self.fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self.dtype)

        def standardized_conv_op(inputs, kernel):
            # Kernel has shape HWIO, normalize over HWI
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
            scale = tf.math.rsqrt(tf.math.maximum(var * self.fan_in, self.__eps__)) * self.gain
            return default_conv_op(inputs, (kernel - mean) * scale)

        self._convolution_op = standardized_conv_op
        self.built = True

    def get_config(self):
        base_config = super(ScaledStandardizedConv2D, self).get_config()
        base_config.update({"eps": self.eps})
        return base_config

class ZeroInitGain(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=(), initializer='zeros', dtype=self.dtype, trainable=True)

    def call(self, inputs):
        return inputs * self.gain

def std_conv2d_with_init(inputs, filters, kernel_size, strides=1, padding="VALID", name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)

    return ScaledStandardizedConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="VALID",
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)

def activation_by_name_with_gamma(inputs, activation="relu", name=None):
    return activation_by_name(inputs, activation, name) * NON_LINEAR_GAMMA.get(activation, 1.0)

def block(
    inputs,
    filters,
    beta=1.0,
    strides=1,
    drop_rate=0,
    alpha=0.2,
    expansion=4,
    se_ratio=0.5,
    group_size=128,
    use_zero_init_gain=True,
    activation="gelu",
    name="",
):
    expanded_filter = filters * expansion
    attn_gain = 2.0
    preact = activation_by_name_with_gamma(inputs, activation, name=name + "preact_") * beta

    if strides > 1 or inputs.shape[-1] != expanded_filter:
        if strides > 1:
            shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_down")(preact)
        else:
            shortcut = preact
        shortcut = std_conv2d_with_init(shortcut, expanded_filter, 1, strides=1, name=name + "shortcut_")
    else:
        shortcut = inputs

    groups = expanded_filter // group_size
    deep = std_conv2d_with_init(preact, filters, 1, strides=1, padding="VALID", name=name + "deep_1_")
    deep = activation_by_name_with_gamma(deep, activation, name=name + "deep_1_")
    deep = std_conv2d_with_init(deep, filters, 3, strides=strides, padding="SAME", groups=groups, name=name + "deep_2_")
    deep = activation_by_name_with_gamma(deep, activation, name=name + "deep_2_")
    deep = std_conv2d_with_init(deep, filters, 3, strides=1, padding="SAME", groups=groups, name=name + "deep_3_") # Extra conv
    deep = activation_by_name_with_gamma(deep, activation, name=name + "deep_3_")
    deep = std_conv2d_with_init(deep, expanded_filter, 1, strides=1, padding="VALID", name=name + "deep_4_")
    if se_ratio > 0:
        deep = se_module(deep, se_ratio=se_ratio, activation=activation, use_bias=True, name=name + "se_")
        deep *= attn_gain

    deep = drop_block(deep, drop_rate)
    if use_zero_init_gain:
        deep = ZeroInitGain(name=name + "deep_gain")(deep)
    deep *= alpha
    return keras.layers.Add(name=name + "add")([shortcut, deep])

def stack(inputs, blocks, filters, betas=1.0, strides=2, stack_drop=0, block_params={}, name=""):
    nn = inputs
    for id in range(blocks):
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        block_drop_rate = stack_drop[id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        beta = betas[id] if isinstance(stack_drop, (list, tuple)) else betas
        nn = block(nn, filters, beta, cur_strides, block_drop_rate, name=block_name, **block_params)
    return nn

def stem(inputs, stem_width, activation="gelu", name=""):
    nn = std_conv2d_with_init(inputs, stem_width // 8, 3, strides=2, padding="same", name=name + "1_")
    nn = activation_by_name_with_gamma(nn, activation, name=name + "1_")
    nn = std_conv2d_with_init(nn, stem_width // 4, 3, strides=1, padding="same", name=name + "2_")
    nn = activation_by_name_with_gamma(nn, activation=activation, name=name + "2_")
    nn = std_conv2d_with_init(nn, stem_width // 2, 3, strides=1, padding="same", name=name + "3_")
    nn = activation_by_name_with_gamma(nn, activation=activation, name=name + "3_")
    nn = std_conv2d_with_init(nn, stem_width, 3, strides=2, padding="same", name=name + "4_")
    return nn

def NormFreeNet(
    num_blocks,
    width_factor=1.0,
    strides=[1, 2, 2, 2],
    out_channels=[64, 128, 256, 512],
    expansion=4,
    stem_width=64,
    input_shape=(224, 224, 3),
    num_classes=1000,
    alpha=0.2,
    se_ratio=0.5,
    group_size=128,
    use_zero_init_gain=True,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    model_name="nfnet",
    kwargs=None
):
    inputs = keras.layers.Input(shape=input_shape)
    stem_width = make_divisible(stem_width * width_factor, 8)
    nn = stem(inputs, stem_width, activation=activation, name="stem_")

    block_params = {  # params same for all blocks
        "alpha": alpha,
        "expansion": expansion,
        "se_ratio": se_ratio,
        "group_size": group_size,
        "use_zero_init_gain": use_zero_init_gain,
        "activation": activation,
    }

    drop_connect_rates = tf.split(tf.linspace(0., drop_connect_rate, sum(num_blocks)), num_blocks)
    drop_connect_rates = [ii.numpy().tolist() for ii in drop_connect_rates]
    beta_list = [(1 + alpha ** 2 * ii) ** -0.5 for ii in range(max(num_blocks) + 1)]
    pre_beta = 1.0
    for id, (num_block, out_channel, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, strides, drop_connect_rates)):
        name = "stack{}_".format(id + 1)
        out_channel = make_divisible(out_channel * width_factor, 8)
        betas = beta_list[:num_block + 1]
        betas[0] = pre_beta
        nn = stack(nn, num_block, out_channel, betas, stride, drop_connect, block_params, name=name)
        pre_beta = betas[-1]

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model
