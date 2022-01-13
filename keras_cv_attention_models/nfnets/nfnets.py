import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import activation_by_name, drop_block, eca_module, se_module, make_divisible, add_pre_post_process

PRETRAINED_DICT = {
    "nfnetf0": {"imagenet": "7f8ee8639d468597de41566ce1b481c7"},
    "nfnetf1": {"imagenet": "f5d298e50996f0a11a8b097e0f890fa2"},
    "nfnetf2": {"imagenet": "3b0f5d6ac33a2833d7d9ee0e02aae4bc"},
    "nfnetf3": {"imagenet": "e28864d7a553cdab9766223994e0a96d"},
    "nfnetf4": {"imagenet": "9a44cb37155f67b88b3900b7c2c9617d"},
    "nfnetf5": {"imagenet": "728d9202661de4d1003c9b149c25461e"},
    "nfnetf6": {"imagenet": "ee4a06b4a543531d72ea5a8a101336ac"},
    "nfnetl0": {"imagenet": "6bd4d11037bf720506aa3d3e12ec4f53"},
    "eca_nfnetl0": {"imagenet": "7789af74226ffee28a0a68cdca6f3737"},
    "eca_nfnetl1": {"imagenet": "cd17a98175825258d32229bc82b744fd"},
    "eca_nfnetl2": {"imagenet": "ca7e0bba4f2d1945d881ffc6e36bed36"},
}

CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
NON_LINEAR_GAMMA = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    swish=1.7881293296813965,  # silu
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


@tf.keras.utils.register_keras_serializable(package="nfnets")
class ScaledStandardizedConv2D(tf.keras.layers.Conv2D):
    """
    Copied from https://github.com/google-research/big_transfer/blob/master/bit_tf2/models.py, Author: Lucas Beyer
    Modified reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """

    def __init__(self, gamma=1.0, eps=1e-5, *args, **kwargs):
        super(ScaledStandardizedConv2D, self).__init__(*args, **kwargs)
        self.eps, self.gamma = eps, gamma

    def build(self, input_shape):
        super(ScaledStandardizedConv2D, self).build(input_shape)
        # Wrap a standardization around the conv OP.
        if hasattr(self, "_convolution_op"):
            default_conv_op = self._convolution_op  # TF < 2.7.0
        else:
            default_conv_op = self.convolution_op  # TF 2.7.0
        self.gain = self.add_weight(name="gain", shape=(self.filters,), initializer="ones", trainable=True, dtype=self.dtype)
        self.fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self._compute_dtype)
        self.__eps__ = tf.cast(self.eps, self._compute_dtype)
        self.__gamma__ = tf.cast(self.gamma, self._compute_dtype)

        def standardized_conv_op(inputs, kernel):
            # Kernel has shape HWIO, normalize over HWI
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
            # print(">>>>", mean.dtype, var.dtype, self.fan_in.dtype, self.__eps__.dtype, self.__gamma__.dtype, self.gain.dtype)
            scale = tf.math.rsqrt(tf.math.maximum(var * self.fan_in, self.__eps__)) * (self.gain * self.__gamma__)
            return default_conv_op(inputs, (kernel - mean) * scale)

        if hasattr(self, "_convolution_op"):
            self._convolution_op = standardized_conv_op  # TF < 2.7.0
        else:
            self.convolution_op = standardized_conv_op  # TF 2.7.0
        self.built = True

    def get_config(self):
        base_config = super(ScaledStandardizedConv2D, self).get_config()
        base_config.update({"eps": self.eps, "gamma": self.gamma})
        return base_config


@tf.keras.utils.register_keras_serializable(package="nfnets")
class ZeroInitGain(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=(), initializer="zeros", dtype=self.dtype, trainable=True)

    def call(self, inputs):
        return inputs * self.gain


def std_conv2d_with_init(inputs, filters, kernel_size, strides=1, padding="VALID", torch_padding=False, gamma=1.0, name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    return ScaledStandardizedConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        gamma=gamma,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def activation_by_name_with_gamma(inputs, activation="gelu", gamma=1.0, name=None):
    nn = activation_by_name(inputs, activation=activation, name=name)
    return nn if gamma == 1.0 else (nn * gamma)


def block(
    inputs,
    filters,
    beta=1.0,
    strides=1,
    drop_rate=0,
    alpha=0.2,
    channel_ratio=0.5,
    se_ratio=0.5,
    group_size=128,
    use_zero_init_gain=True,
    torch_padding=False,
    attn_type="se",
    conv_gamma=1.0,
    act_gamma=1.0,
    activation="gelu",
    name="",
):
    hidden_filter = int(filters * channel_ratio)
    attn_gain = 2.0
    # print(f">>>> {beta = }")
    preact = activation_by_name_with_gamma(inputs, activation, gamma=act_gamma, name=name + "preact_") * beta

    if strides > 1 or inputs.shape[-1] != filters:
        if strides > 1:
            shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_down")(preact)
        else:
            shortcut = preact
        shortcut = std_conv2d_with_init(shortcut, filters, 1, strides=1, gamma=conv_gamma, name=name + "shortcut_")
    else:
        shortcut = inputs

    groups = hidden_filter // group_size
    conv_params_3 = {"kernel_size": 3, "padding": "SAME", "torch_padding": torch_padding, "gamma": conv_gamma}
    deep = std_conv2d_with_init(preact, hidden_filter, 1, strides=1, gamma=conv_gamma, name=name + "deep_1_")
    deep = activation_by_name_with_gamma(deep, activation, gamma=act_gamma, name=name + "deep_1_")
    deep = std_conv2d_with_init(deep, hidden_filter, strides=strides, **conv_params_3, groups=groups, name=name + "deep_2_")
    deep = activation_by_name_with_gamma(deep, activation, gamma=act_gamma, name=name + "deep_2_")
    deep = std_conv2d_with_init(deep, hidden_filter, strides=1, **conv_params_3, groups=groups, name=name + "deep_3_")  # Extra conv
    deep = activation_by_name_with_gamma(deep, activation, gamma=act_gamma, name=name + "deep_3_")
    deep = std_conv2d_with_init(deep, filters, 1, strides=1, gamma=conv_gamma, name=name + "deep_4_")
    if se_ratio > 0 and attn_type == "se":
        deep = se_module(deep, se_ratio=se_ratio, activation="relu", use_bias=True, name=name + "se_")
        deep *= attn_gain
    elif attn_type == "eca":
        deep = eca_module(deep, name=name + "eca_")
        deep *= attn_gain

    deep = drop_block(deep, drop_rate)
    if use_zero_init_gain:
        deep = ZeroInitGain(name=name + "deep_gain")(deep)
    deep *= alpha
    return keras.layers.Add(name=name + "output")([shortcut, deep])


def stack(inputs, blocks, filters, betas=1.0, strides=2, stack_drop=0, block_params={}, name=""):
    nn = inputs
    for id in range(blocks):
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        block_drop_rate = stack_drop[id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        beta = betas[id] if isinstance(stack_drop, (list, tuple)) else betas
        nn = block(nn, filters, beta, cur_strides, block_drop_rate, name=block_name, **block_params)
    return nn


def stem(inputs, stem_width, activation="gelu", torch_padding=False, conv_gamma=1.0, act_gamma=1.0, name=""):
    conv_params = {"kernel_size": 3, "padding": "SAME", "torch_padding": torch_padding, "gamma": conv_gamma}
    nn = std_conv2d_with_init(inputs, stem_width // 8, strides=2, **conv_params, name=name + "1_")
    nn = activation_by_name_with_gamma(nn, activation, gamma=act_gamma, name=name + "1_")
    nn = std_conv2d_with_init(nn, stem_width // 4, strides=1, **conv_params, name=name + "2_")
    nn = activation_by_name_with_gamma(nn, activation, gamma=act_gamma, name=name + "2_")
    nn = std_conv2d_with_init(nn, stem_width // 2, strides=1, **conv_params, name=name + "3_")
    nn = activation_by_name_with_gamma(nn, activation, gamma=act_gamma, name=name + "3_")
    nn = std_conv2d_with_init(nn, stem_width, strides=2, **conv_params, name=name + "4_")
    return nn


def NormFreeNet(
    num_blocks,
    attn_type="se",
    stem_width=128,
    out_channels=[256, 512, 1536, 1536],
    channel_ratio=0.5,
    num_features_factor=2,
    strides=[1, 2, 2, 2],
    input_shape=(224, 224, 3),
    num_classes=1000,
    se_ratio=0.5,
    group_size=128,
    use_zero_init_gain=True,
    torch_padding=False,
    gamma_in_act=True,
    alpha=0.2,
    width_factor=1.0,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained="imagenet",
    model_name="nfnet",
    kwargs=None,
):
    if gamma_in_act:
        # activation.split("/")[0] for supporting `gelu/app`
        conv_gamma, act_gamma = 1.0, NON_LINEAR_GAMMA.get(activation.split("/")[0], 1.0)
    else:
        act_gamma, conv_gamma = 1.0, NON_LINEAR_GAMMA.get(activation.split("/")[0], 1.0)

    inputs = keras.layers.Input(shape=input_shape)
    stem_width = make_divisible(stem_width * width_factor, 8)
    nn = stem(inputs, stem_width, activation=activation, torch_padding=torch_padding, conv_gamma=conv_gamma, act_gamma=act_gamma, name="stem_")

    block_params = {  # params same for all blocks
        "alpha": alpha,
        "channel_ratio": channel_ratio,
        "se_ratio": se_ratio,
        "group_size": group_size,
        "use_zero_init_gain": use_zero_init_gain,
        "torch_padding": torch_padding,
        "attn_type": attn_type,
        "conv_gamma": conv_gamma,
        "act_gamma": act_gamma,
        "activation": activation,
    }

    drop_connect_rates = tf.split(tf.linspace(0.0, drop_connect_rate, sum(num_blocks)), num_blocks)
    drop_connect_rates = [ii.numpy().tolist() for ii in drop_connect_rates]
    beta_list = [(1 + alpha ** 2 * ii) ** -0.5 for ii in range(max(num_blocks) + 1)]
    pre_beta = 1.0
    for id, (num_block, out_channel, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, strides, drop_connect_rates)):
        name = "stack{}_".format(id + 1)
        out_channel = make_divisible(out_channel * width_factor, 8)
        betas = beta_list[: num_block + 1]
        betas[0] = pre_beta
        nn = stack(nn, num_block, out_channel, betas, stride, drop_connect, block_params, name=name)
        pre_beta = betas[-1]

    if num_features_factor > 0:
        output_conv_filter = make_divisible(num_features_factor * out_channels[-1] * width_factor, 8)
        nn = std_conv2d_with_init(nn, output_conv_filter, 1, gamma=conv_gamma, name="post_")
    nn = activation_by_name_with_gamma(nn, activation, gamma=act_gamma, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="nfnets", pretrained=pretrained)
    return model


def NFNetF0(input_shape=(256, 256, 3), num_classes=1000, activation="gelu", dropout=0.2, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[1, 2, 6, 3], model_name="nfnetf0", **locals(), **kwargs)


def NFNetF1(input_shape=(320, 320, 3), num_classes=1000, activation="gelu", dropout=0.3, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[2, 4, 12, 6], model_name="nfnetf1", **locals(), **kwargs)


def NFNetF2(input_shape=(352, 352, 3), num_classes=1000, activation="gelu", dropout=0.4, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[3, 6, 18, 9], model_name="nfnetf2", **locals(), **kwargs)


def NFNetF3(input_shape=(416, 416, 3), num_classes=1000, activation="gelu", dropout=0.4, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[4, 8, 24, 12], model_name="nfnetf3", **locals(), **kwargs)


def NFNetF4(input_shape=(512, 512, 3), num_classes=1000, activation="gelu", dropout=0.5, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[5, 10, 30, 15], model_name="nfnetf4", **locals(), **kwargs)


def NFNetF5(input_shape=(544, 544, 3), num_classes=1000, activation="gelu", dropout=0.5, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[6, 12, 36, 18], model_name="nfnetf5", **locals(), **kwargs)


def NFNetF6(input_shape=(576, 576, 3), num_classes=1000, activation="gelu", dropout=0.5, pretrained="imagenet", **kwargs):
    return NormFreeNet(num_blocks=[7, 14, 42, 21], model_name="nfnetf6", **locals(), **kwargs)


def NFNetF7(input_shape=(608, 608, 3), num_classes=1000, activation="gelu", dropout=0.5, pretrained=None, **kwargs):
    return NormFreeNet(num_blocks=[8, 16, 48, 24], model_name="nfnetf7", **locals(), **kwargs)


def NormFreeNet_Light(channel_ratio=0.25, group_size=64, torch_padding=True, use_zero_init_gain=False, gamma_in_act=False, **kwargs):
    kwargs.pop("kwargs", None)
    return NormFreeNet(**locals(), **kwargs)


def NFNetL0(input_shape=(288, 288, 3), num_classes=1000, activation="swish", dropout=0.2, pretrained="imagenet", **kwargs):
    num_blocks = [1, 2, 6, 3]
    num_features_factor = 1.5
    se_ratio = 0.25
    return NormFreeNet_Light(model_name="nfnetl0", **locals(), **kwargs)


def ECA_NFNetL0(input_shape=(288, 288, 3), num_classes=1000, activation="swish", dropout=0.2, pretrained="imagenet", **kwargs):
    num_blocks = [1, 2, 6, 3]
    num_features_factor = 1.5
    attn_type = "eca"
    return NormFreeNet_Light(model_name="eca_nfnetl0", **locals(), **kwargs)


def ECA_NFNetL1(input_shape=(320, 320, 3), num_classes=1000, activation="swish", dropout=0.2, pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 12, 6]
    attn_type = "eca"
    return NormFreeNet_Light(model_name="eca_nfnetl1", **locals(), **kwargs)


def ECA_NFNetL2(input_shape=(384, 384, 3), num_classes=1000, activation="swish", dropout=0.2, pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 18, 9]
    attn_type = "eca"
    return NormFreeNet_Light(model_name="eca_nfnetl2", **locals(), **kwargs)


def ECA_NFNetL3(input_shape=(448, 448, 3), num_classes=1000, activation="swish", dropout=0.2, pretrained=None, **kwargs):
    num_blocks = [4, 8, 24, 12]
    attn_type = "eca"
    return NormFreeNet_Light(model_name="eca_nfnetl3", **locals(), **kwargs)
