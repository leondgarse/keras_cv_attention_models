import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import layer_norm, conv2d_no_bias, activation_by_name, add_pre_post_process


PRETRAINED_DICT = {
    "coat_lite_tiny": {"imagenet": "7738d1efb345d17f7569929330a2cf7d"},
    "coat_lite_mini": {"imagenet": "9ad2fa037addee382e70c6fac1941a68"},
    "coat_lite_small": {"imagenet": "0c8012cfba5b1d1b97305770587730ff"},
    "coat_tiny": {"imagenet": "0b20a82b7f82a3d73cca9fb5b66db8fb"},
    "coat_mini": {"imagenet": "883a0c3083b82f19f1245572ef068311"},
}


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = keras.layers.Dense(hidden_dim, name=name + "dense_0")(inputs)
    nn = activation_by_name(nn, activation, name=name and name + activation)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "dense_1")(nn)
    return nn


@keras.utils.register_keras_serializable(package="coat")
class ConvPositionalEncoding(keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(ConvPositionalEncoding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
        self.supports_masking = False

    def build(self, input_shape):
        self.height = self.width = int(tf.math.sqrt(float(input_shape[1] - 1)))  # assume hh == ww
        self.channel = input_shape[-1]
        # Conv2D with goups=self.channel
        self.dconv = keras.layers.DepthwiseConv2D(
            self.kernel_size,
            strides=1,
            padding="VALID",
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


@keras.utils.register_keras_serializable(package="coat")
class ConvRelativePositionalEncoding(keras.layers.Layer):
    def __init__(self, head_splits=[2, 3, 3], head_kernel_size=[3, 5, 7], **kwargs):
        super(ConvRelativePositionalEncoding, self).__init__(**kwargs)
        self.head_splits, self.head_kernel_size = head_splits, head_kernel_size
        self.supports_masking = False

    def build(self, query_shape):
        # print(query_shape)
        self.height = self.width = int(tf.math.sqrt(float(query_shape[2] - 1)))  # assume hh == ww
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
                    name=name_scope if self.name is None else self.name + name_scope,
                )
                # print(query_shape, [None, self.height, self.width, int(head_split * self.query_dim)])
                dconv.build([None, self.height, self.width, int(head_split * self.query_dim)])
            pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
            self.dconvs.append(dconv)
            self.pads.append(pad)

    def call(self, query, value, **kwargs):
        img_token_q, img_token_v = query[:, :, 1:, :], value[:, :, 1:, :]

        img_token_v = tf.transpose(img_token_v, [0, 2, 1, 3])  # [batch, blocks, num_heads, query_dim]
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


@keras.utils.register_keras_serializable(package="coat")
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
    key_dim = dim // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))

    qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name=name + "qkv")(inputs)
    qkv = keras.layers.Reshape([blocks, 3, num_heads, key_dim])(qkv)
    qq, kk, vv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [qkv, batch, num_heads, blocks, key_dim]
    # print(f">>>> {qkv.shape = }, {qq.shape = }, {kk.shape = }, {vv.shape = }")

    # Factorized attention.
    # kk = tf.nn.softmax(kk, axis=2)  # On `blocks` dimension
    kk = keras.layers.Softmax(axis=2, name=name and name + "attention_scores")(kk)  # On `blocks` dimension
    # attn = tf.matmul(kk, vv, transpose_a=True)  # 'b h n k, b h n v -> b h k v', [batch, num_heads, key_dim, key_dim]
    # factor_att = tf.matmul(qq, attn)    # 'b h n k, b h k v -> b h n v', [batch, num_heads, blocks, key_dim]
    attn = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_a=True))([kk, vv])
    factor_att = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([qq, attn])

    # Convolutional relative position encoding.
    crpe_out = shared_crpe(qq, vv) if shared_crpe is not None else ConvRelativePositionalEncoding(name=name + "crpe")(qq, vv)

    # Merge and reshape.
    nn = keras.layers.Add()([factor_att * qk_scale, crpe_out])
    nn = keras.layers.Permute([2, 1, 3])(nn)
    nn = keras.layers.Reshape([blocks, dim])(nn)

    # Output projection.
    nn = keras.layers.Dense(dim, name=name + "out")(nn)
    # Drop
    return nn


def __cpe_norm_crpe__(inputs, shared_cpe=None, shared_crpe=None, num_heads=8, name=""):
    cpe_out = shared_cpe(inputs) if shared_cpe is not None else ConvPositionalEncoding(name=name + "cpe")(inputs)  # shared
    nn = layer_norm(cpe_out, name=name + "norm1")
    crpe_out = factor_attention_conv_relative_positional_encoding(nn, shared_crpe=shared_crpe, num_heads=num_heads, name=name + "factoratt_crpe_")
    return cpe_out, crpe_out


def __res_mlp_block__(cpe_out, crpe_out, mlp_ratio=4, drop_rate=0, activation="gelu", name=""):
    if drop_rate > 0:
        crpe_out = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_1")(crpe_out)
    cpe_crpe = keras.layers.Add()([cpe_out, crpe_out])

    # MLP
    nn = layer_norm(cpe_crpe, name=name + "norm2")
    nn = mlp_block(nn, nn.shape[-1] * mlp_ratio, activation=activation, name=name + "mlp_")
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_2")(nn)
    return keras.layers.Add(name=name + "output")([cpe_crpe, nn])


def serial_block(inputs, embed_dim, shared_cpe=None, shared_crpe=None, num_heads=8, mlp_ratio=4, drop_rate=0, activation="gelu", name=""):
    cpe_out, crpe_out = __cpe_norm_crpe__(inputs, shared_cpe, shared_crpe, num_heads, name=name)
    out = __res_mlp_block__(cpe_out, crpe_out, mlp_ratio, drop_rate, activation=activation, name=name)
    return out


def resample(image, class_token=None, factor=1):
    out_hh, out_ww = int(image.shape[1] * factor), int(image.shape[2] * factor)
    out_image = tf.cast(tf.image.resize(image, [out_hh, out_ww], method="bilinear"), image.dtype)
    # if factor > 1:
    #     out_image = keras.layers.UpSampling2D(factor, interpolation='bilinear')(image)
    # elif factor == 1:
    #     out_image = image
    # else:
    #     size = int(1 / factor)
    #     out_image = keras.layers.AvgPool2D(size, strides=size)(image)

    if class_token is not None:
        out_image = tf.reshape(out_image, [-1, out_hh * out_ww, out_image.shape[-1]])
        return tf.concat([class_token, out_image], axis=1)
    else:
        return out_image


def parallel_block(inputs, shared_cpes=None, shared_crpes=None, num_heads=8, mlp_ratios=[], drop_rate=0, activation="gelu", name=""):
    # Conv-Attention.
    cpe_outs, crpe_outs, crpe_images = [], [], []
    for id, (xx, shared_cpe, shared_crpe) in enumerate(zip(inputs[1:], shared_cpes[1:], shared_crpes[1:])):
        cur_name = name + "{}_".format(id + 2)
        cpe_out, crpe_out = __cpe_norm_crpe__(xx, shared_cpe, shared_crpe, num_heads, name=cur_name)
        cpe_outs.append(cpe_out)
        crpe_outs.append(crpe_out)
        hh = ww = int(tf.math.sqrt(float(crpe_out.shape[1] - 1)))  # assume hh == ww
        crpe_images.append(tf.reshape(crpe_out[:, 1:, :], [-1, hh, ww, crpe_out.shape[-1]]))
        # print(f">>>> {crpe_out.shape = }, {crpe_images[-1].shape = }")
    crpe_stack = [  # [[None, 28, 28, 152], [None, 14, 14, 152], [None, 7, 7, 152]]
        crpe_outs[0] + resample(crpe_images[1], crpe_outs[1][:, :1], factor=2) + resample(crpe_images[2], crpe_outs[2][:, :1], factor=4),
        crpe_outs[1] + resample(crpe_images[2], crpe_outs[2][:, :1], factor=2) + resample(crpe_images[0], crpe_outs[0][:, :1], factor=1 / 2),
        crpe_outs[2] + resample(crpe_images[1], crpe_outs[1][:, :1], factor=1 / 2) + resample(crpe_images[0], crpe_outs[0][:, :1], factor=1 / 4),
    ]

    # MLP
    outs = []
    for id, (cpe_out, crpe_out, mlp_ratio) in enumerate(zip(cpe_outs, crpe_stack, mlp_ratios[1:])):
        cur_name = name + "{}_".format(id + 2)
        out = __res_mlp_block__(cpe_out, crpe_out, mlp_ratio, drop_rate, activation=activation, name=cur_name)
        outs.append(out)
    return inputs[:1] + outs  # inputs[0] directly out


def patch_embed(inputs, embed_dim, patch_size=2, name=""):
    if len(inputs.shape) == 3:
        height = width = int(tf.math.sqrt(float(inputs.shape[1])))  # assume hh == ww
        inputs = keras.layers.Reshape([height, width, inputs.shape[-1]])(inputs)
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=True, name=name)  # Try with Conv1D
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], nn.shape[-1]])(nn)  # flatten(2)
    nn = layer_norm(nn, name=name)
    return nn


def CoaT(
    serial_depths,
    embed_dims,
    mlp_ratios,
    parallel_depth=0,
    patch_size=4,
    num_heads=8,
    head_splits=[2, 3, 3],
    head_kernel_size=[3, 5, 7],
    use_shared_cpe=True,
    use_shared_crpe=True,
    out_features=None,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="coat",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    # serial blocks
    nn = inputs
    classfier_outs = []
    shared_cpes = []
    shared_crpes = []
    for sid, (depth, embed_dim, mlp_ratio) in enumerate(zip(serial_depths, embed_dims, mlp_ratios)):
        name = "serial{}_".format(sid + 1)
        patch_size = patch_size if sid == 0 else 2
        # print(f">>>> {nn.shape = }")
        nn = patch_embed(nn, embed_dim, patch_size=patch_size, name=name + "patch_")
        nn = ClassToken(name=name + "class_token")(nn)
        shared_cpe = ConvPositionalEncoding(kernel_size=3, name="cpe_" + str(sid + 1)) if use_shared_cpe else None
        shared_crpe = ConvRelativePositionalEncoding(head_splits, head_kernel_size, name="crpe_" + str(sid + 1)) if use_shared_crpe else None
        for bid in range(depth):
            block_name = name + "block{}_".format(bid + 1)
            nn = serial_block(nn, embed_dim, shared_cpe, shared_crpe, num_heads, mlp_ratio, activation=activation, name=block_name)
        classfier_outs.append(nn)
        shared_cpes.append(shared_cpe)
        shared_crpes.append(shared_crpe)
        nn = nn[:, 1:, :]  # remove class token

    # Parallel blocks.
    for pid in range(parallel_depth):
        name = "parallel{}_".format(pid + 1)
        classfier_outs = parallel_block(classfier_outs, shared_cpes, shared_crpes, num_heads, mlp_ratios, activation=activation, name=name)

    if out_features is not None:  # Return intermediate features (for down-stream tasks).
        nn = [classfier_outs[id][:, 1:, :] for id in out_features]
    elif parallel_depth == 0:  # Lite model, only serial blocks, Early return.
        nn = layer_norm(classfier_outs[-1], name="out_")[:, 0]
    else:
        nn = [layer_norm(xx, name="out_{}_".format(id + 1))[:, :1, :] for id, xx in enumerate(classfier_outs[1:])]
        nn = keras.layers.Concatenate(axis=1)(nn)
        nn = keras.layers.Permute([2, 1])(nn)
        nn = keras.layers.Conv1D(1, 1, name="aggregate")(nn)[:, :, 0]

    if out_features is None and num_classes > 0:
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="coat", pretrained=pretrained)
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


def CoaTLiteTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_tiny"], **locals(), model_name="coat_lite_tiny", **kwargs)


def CoaTLiteMini(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_mini"], **locals(), model_name="coat_lite_mini", **kwargs)


def CoaTLiteSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["lite_small"], **locals(), model_name="coat_lite_small", **kwargs)


def CoaTTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["tiny"], **locals(), model_name="coat_tiny", **kwargs)


def CoaTMini(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**BLOCK_CONFIGS["mini"], **locals(), model_name="coat_mini", **kwargs)
