import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import os

# from keras_cv_attention_models import attention_layers

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-6
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=None):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if activation:
        nn = keras.layers.Activation(activation=activation, name=name and name + activation)(nn)
    return nn


def layer_norm(inputs, name=None):
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=BATCH_NORM_EPSILON, name=name and name + "ln")(inputs)


def conv2d_with_init(inputs, filters, kernel_size, strides=1, padding="VALID", name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def patch_embed(inputs, embed_dim, patch_size=2, name=None):
    if len(inputs.shape) == 3:
        height = width = int(tf.math.sqrt(float(inputs.shape[1]))) # assume hh == ww
        inputs = keras.layers.Reshape([height, width, inputs.shape[-1]])(inputs)
    nn = conv2d_with_init(inputs, embed_dim, kernel_size=patch_size, strides=patch_size, name=name) # Try with Conv1D
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], nn.shape[-1]])(nn)  # flatten(2)
    nn = layer_norm(nn, name=name)
    return nn


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = keras.layers.Dense(hidden_dim, name=name + "Dense_0")(inputs)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "Dense_1")(nn)
    return nn

class ConvPositionalEncoding(keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(ConvPositionalEncoding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
        self.supports_masking = False

    def build(self, input_shape):
        self.height = self.width = int(tf.math.sqrt(float(input_shape[1] - 1))) # assume hh == ww
        self.channel = input_shape[-1]
        # Conv2D with goups=self.channel
        self.dconv = keras.layers.DepthwiseConv2D(
            self.kernel_size,
            strides=1,
            padding="VALID",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name and self.name + "depth_conv",
        )
        self.dconv.build([None, self.height, self.width, self.channel])
        super(ConvPositionalEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        cls_token, img_token = inputs[:, :1], inputs[:, 1:]
        img_token = tf.reshape(img_token, [-1, self.height, self.width, self.channel])
        nn = self.dconv(tf.pad(img_token, self.pad)) + img_token
        nn = tf.reshape(nn, [-1, self.height * self.width, self.channel])
        return tf.concat([cls_token, nn], axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(ConvPositionalEncoding, self).get_config()
        base_config.update({"kernel_size": self.kernel_size})
        return base_config

class ConvRelativePositionalEncoding(keras.layers.Layer):
    def __init__(self, head_splits=[2, 3, 3], head_kernel_size=[3, 5, 7], **kwargs):
        super(ConvRelativePositionalEncoding, self).__init__(**kwargs)
        self.head_splits, self.head_kernel_size = head_splits, head_kernel_size
        self.supports_masking = False

    def build(self, query_shape):
        # print(query_shape)
        self.height = self.width = int(tf.math.sqrt(float(query_shape[2] - 1))) # assume hh == ww
        self.num_heads, self.query_dim = query_shape[1], query_shape[-1]
        self.channel_splits = [ii * self.query_dim for ii in self.head_splits]

        self.dconvs = []
        self.pads = []
        for id, (head_split, kernel_size) in enumerate(zip(self.head_splits, self.head_kernel_size)):
            name_scope = "depth_conv_" + str(id + 1)
            with tf.name_scope(name_scope) as scope:
                dconv = keras.layers.DepthwiseConv2D(
                    kernel_size,
                    strides=1,
                    padding="VALID",
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=name_scope if self.name is None else self.name + name_scope,
                )
                # print(query_shape, [None, self.height, self.width, int(head_split * self.query_dim)])
                dconv.build([None, self.height, self.width, int(head_split * self.query_dim)])
            pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
            self.dconvs.append(dconv)
            self.pads.append(pad)

    def call(self, query, value, **kwargs):
        img_token_q, img_token_v = query[:, :, 1:, :], value[:, :, 1:, :]

        img_token_v = tf.transpose(img_token_v, [0, 2, 1, 3])
        img_token_v = tf.reshape(img_token_v, [-1, self.height, self.width, self.num_heads * self.query_dim])
        split_values = tf.split(img_token_v, self.channel_splits, axis=-1)
        nn = [dconv(tf.pad(split_value, pad)) for split_value, dconv, pad in zip(split_values, self.dconvs, self.pads)]
        nn = tf.concat(nn, axis=-1)
        conv_v_img = tf.reshape(nn, [-1, self.height * self.width, self.num_heads, self.query_dim])
        conv_v_img = tf.transpose(conv_v_img, [0, 2, 1, 3])

        EV_hat_img = img_token_q * conv_v_img
        return tf.pad(EV_hat_img, [[0, 0], [0, 0], [1, 0], [0, 0]])

    def get_config(self):
        base_config = super(ConvRelativePositionalEncoding, self).get_config()
        base_config.update({"head_splits": self.head_splits, "head_kernel_size": self.head_kernel_size})
        return base_config

class ClassToken(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.token_init = tf.initializers.TruncatedNormal(stddev=0.2)
        self.supports_masking = False

    def build(self, input_shape):
        self.class_tokens = self.add_weight(name="tokens", shape=(1, 1, input_shape[-1]), initializer=self.token_init, trainable=True)
        super(ClassToken, self).build(input_shape)

    def call(self, inputs, **kwargs):
        class_tokens = tf.tile(self.class_tokens, [tf.shape(inputs)[0], 1, 1])
        return tf.concat([class_tokens, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

def factor_attention_conv_relative_positional_encoding(inputs, shared_crpe=None, num_heads=8, qkv_bias=True, name=""):
    blocks, dim = inputs.shape[1], inputs.shape[-1]
    qk_scale = 1.0 / tf.math.sqrt(float(dim // num_heads))

    qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name=name + "qkv")(inputs)
    qkv = keras.layers.Reshape([blocks, 3, num_heads, dim // num_heads])(qkv)
    qq, kk, vv = tf.transpose(qkv, [2, 0, 3, 1, 4])

    # Factorized attention.
    kk = tf.nn.softmax(kk, axis=2)
    attn = tf.matmul(kk, vv, transpose_a=True)
    factor_att = tf.matmul(qq, attn)
    attn = keras.layers.Lambda

    # Convolutional relative position encoding.
    crpe_out = shared_crpe(qq, vv) if shared_crpe is not None else ConvRelativePositionalEncoding(name=name + "crpe")(qq, vv)

    # Merge and reshape.
    nn = keras.layers.Add()([factor_att * qk_scale, crpe_out])
    nn = keras.layers.Permute([2, 1, 3])(nn)
    nn = keras.layers.Reshape([blocks, dim])(nn)

    # Output projection.
    nn = keras.layers.Dense(dim, name=name + "factor_atten_out")(nn)
    # Drop
    return nn

def serial_block(inputs, embed_dim, shared_cpe=None, shared_crpe=None, num_heads=8, mlp_ratio=4, drop_rate=0, activation="relu", name=""):
    nn = shared_cpe(inputs) if shared_cpe else ConvPositionalEncoding(name=name + "cpe")(inputs)  # shared
    cur = layer_norm(nn, name=name)
    cur = factor_attention_conv_relative_positional_encoding(cur, shared_crpe=shared_crpe, num_heads=num_heads, name=name)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_1")(cur)
    nn = keras.layers.Add()([nn, cur])

    # MLP
    cur = layer_norm(nn, name=name + "mlp_")
    cur = mlp_block(cur, cur.shape[-1] * mlp_ratio, activation=activation, name=name+ "mlp_")
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_2")(cur)
    nn = keras.layers.Add()([nn, cur])
    return nn

def parallel_block(inputs, embed_dim, shared_cpes=None, shared_crpes=None, num_heads=8, mlp_ratios=[], drop_rate=0, activation="relu", name=""):
    return inputs

def CoaT(
    serial_depths,
    embed_dims,
    mlp_ratios,
    parallel_depth=0,
    patch_size=4,
    num_heads=8,
    head_splits=[2, 3, 3],
    head_kernel_size=[3, 5, 7],
    use_share_cpe=True,
    use_share_crpe=True,
    out_features=None,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="coat",
    **kwargs,
):
    inputs = keras.layers.Input(input_shape)

    # serial blocks
    nn = inputs
    classfier_outs = []
    shared_cpes = []
    shared_crpes = []
    for sid, (depth, embed_dim, mlp_ratio) in enumerate(zip(serial_depths, embed_dims, mlp_ratios)):
        name = "serial_{}_".format(sid + 1)
        patch_size = patch_size if sid == 0 else 2
        # print(f">>>> {nn.shape = }")
        nn = patch_embed(nn, embed_dim, patch_size=patch_size, name=name + "patch")
        nn = ClassToken(name=name + "class_token")(nn)
        shared_cpe = ConvPositionalEncoding(kernel_size=3, name="cpe_" + str(sid + 1)) if use_share_cpe else None
        shared_crpe = ConvRelativePositionalEncoding(head_splits, head_kernel_size, name="crpe_" + str(sid + 1)) if use_share_crpe else None
        for bid in range(depth):
            block_name = name + "{}_".format(bid + 1)
            nn = serial_block(nn, embed_dim, shared_cpe, shared_crpe, num_heads, mlp_ratio, activation=activation, name=block_name)
        classfier_outs.append(nn)
        shared_cpes.append(shared_cpe)
        shared_crpes.append(shared_crpe)
        nn = nn[:, 1:, :]   # remove class token

    # Parallel blocks.
    for ipid in range(parallel_depth):
        classfier_outs = parallel_block(classfier_outs, embed_dims, shared_cpes, shared_crpes, num_heads, mlp_ratios, activation=activation, name="parallel_")

    if out_features is None:
        nn = [layer_norm(xx, name="out_ln_{}".format(id + 1))[:, :1, :] for id, xx in enumerate(classfier_outs[1:])]
        nn = keras.layers.Concatenate(axis=-1)(nn)
        nn = keras.layers.Permute([2, 1])(nn)
        nn = keras.layers.Conv1D(1, 1, name="out_conv1d")(nn)[:, :, 0]
    else:  # Return intermediate features (for down-stream tasks).
        nn = [xx[:, 1:, :] for xx in classfier_outs]

    if out_features is None and num_classes > 0:
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


BLOCK_CONFIGS = {
    "lite_tiny": {
        "serial_depths": [2, 2, 2, 2],
        "embed_dims": [64, 128, 256, 320],
        "mlp_ratios": [8, 8, 4, 4],
        "parallel_depth": 0,
        "patch_size": 4,
        "num_heads": 8,
    },
    "lite_mini": {
        "serial_depths": [2, 2, 2, 2],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "parallel_depth": 0,
        "patch_size": 4,
        "num_heads": 8,
    },
    "lite_small": {
        "serial_depths": [3, 4, 6, 3],
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "parallel_depth": 0,
        "patch_size": 4,
        "num_heads": 8,
    },
    "tiny": {
        "serial_depths": [2, 2, 2, 2],
        "embed_dims": [152, 152, 152, 152],
        "mlp_ratios": [4, 4, 4, 4],
        "parallel_depth": 6,
        "patch_size": 4,
        "num_heads": 8,
    },
    "mini": {
        "serial_depths": [2, 2, 2, 2],
        "embed_dims": [152, 216, 216, 216],
        "mlp_ratios": [4, 4, 4, 4],
        "parallel_depth": 6,
        "patch_size": 4,
        "num_heads": 8,
    },
}

def CoaTLiteTiny(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_tiny"], **locals(), model_name="coat_lite_tiny", **kwargs)

def CoaTLiteMini(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_mini"], **locals(), model_name="coat_lite_mini", **kwargs)

def CoaTLiteSmall(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_small"], **locals(), model_name="coat_lite_small", **kwargs)

def CoaTTiny(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["tiny"], **locals(), model_name="coat_tiny", **kwargs)

def CoaTMini(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["mini"], **locals(), model_name="coat_mini", **kwargs)
