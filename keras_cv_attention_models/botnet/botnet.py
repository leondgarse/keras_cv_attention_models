"""
A Keras version of `botnet`.
Original TensorFlow version: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights_with_mismatch
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

PRETRAINED_DICT = {
    "botnet50": {"imagenet": "b221b45ca316166fc858fda1cf4fd946"},
}


@tf.keras.utils.register_keras_serializable(package="botnet")
class RelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, position_height=0, position_width=0, use_absolute_pos=False, **kwargs):
        super(RelativePositionalEmbedding, self).__init__(**kwargs)
        self.position_height = position_height
        self.position_width = position_width if position_width > 0 else position_height
        self.use_absolute_pos = use_absolute_pos

    def build(self, input_shape):
        _, num_heads, height, width, key_dim = input_shape
        self.position_height = self.position_height if self.position_height > height else height
        self.position_width = self.position_width if self.position_width > width else width
        self.key_dim = key_dim
        stddev = key_dim ** -0.5

        if self.use_absolute_pos:
            hh_shape = (key_dim, self.position_height)
            ww_shape = (key_dim, self.position_width)
        else:
            hh_shape = (key_dim, 2 * self.position_height - 1)
            ww_shape = (key_dim, 2 * self.position_width - 1)

        initializer = tf.random_normal_initializer(stddev=stddev)
        self.pos_emb_h = self.add_weight(name="r_height", shape=hh_shape, initializer=initializer, trainable=True)
        self.pos_emb_w = self.add_weight(name="r_width", shape=ww_shape, initializer=initializer, trainable=True)

    def get_config(self):
        base_config = super(RelativePositionalEmbedding, self).get_config()
        base_config.update({"position_height": self.position_height, "position_width": self.position_width, "use_absolute_pos": self.use_absolute_pos})
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, height, width, 2 * pos_dim - 1]
        Output: [bs, heads, height, width, pos_dim]
        """
        _, heads, hh, ww, dim = rel_pos.shape  # [bs, heads, height, width, 2 * width - 1]
        pos_dim = (dim + 1) // 2
        if pos_dim == 1:
            return rel_pos
        if ww == 1:
            return rel_pos[..., -pos_dim:]
        full_rank_gap = pos_dim - ww
        # [bs, heads, height, width * (2 * pos_dim - 1)] --> [bs, heads, height, width * (2 * pos_dim - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * dim])[:, :, :, ww - 1 : -1]
        # [bs, heads, height, width, 2 * (pos_dim - 1)] --> [bs, heads, height, width, pos_dim]
        # print(f">>>> {full_rank_gap = }, {flat_x.shape = }")
        return tf.reshape(flat_x, [-1, heads, hh, ww, 2 * (pos_dim - 1)])[:, :, :, :, full_rank_gap : pos_dim + full_rank_gap]

    def relative_logits(self, inputs):
        query_w = inputs  # e.g.: [1, 4, 14, 16, 128], [bs, heads, hh, ww, dims]
        rel_logits_w = tf.matmul(query_w, self.pos_emb_w)  # [1, 4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)  # [1, 4, 14, 16, 16]

        query_h = tf.transpose(inputs, [0, 1, 3, 2, 4])  # [1, 4, 16, 14, 128], [bs, heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(query_h, self.pos_emb_h)  # [1, 4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [1, 4, 16, 14, 14]
        rel_logits_h = tf.transpose(rel_logits_h, [0, 1, 3, 2, 4])  # [1, 4, 14, 16, 14], transpose back

        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)  # [1, 4, 14, 16, 14, 16]

    def absolute_logits(self, inputs):
        pos_emb = tf.expand_dims(self.pos_emb_h, 2) + tf.expand_dims(self.pos_emb_w, 1)
        abs_logits = tf.einsum("bxyhd,dpq->bhxypq", inputs, pos_emb)
        return abs_logits

    def call(self, inputs):
        return self.absolute_logits(inputs) if self.use_absolute_pos else self.relative_logits(inputs)

    def load_resized_pos_emb(self, source_layer):
        # For input 224 --> [128, 27], convert to 480 --> [128, 30]
        image_like_w = tf.expand_dims(tf.transpose(source_layer.pos_emb_w, [1, 0]), 0)
        resize_w = tf.image.resize(image_like_w, (1, self.pos_emb_w.shape[1]))[0]
        self.pos_emb_w.assign(tf.transpose(resize_w, [1, 0]))

        image_like_h = tf.expand_dims(tf.transpose(source_layer.pos_emb_h, [1, 0]), 0)
        resize_h = tf.image.resize(image_like_h, (1, self.pos_emb_h.shape[1]))[0]
        self.pos_emb_h.assign(tf.transpose(resize_h, [1, 0]))


def mhsa_with_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, relative=True, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(float(key_dim))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim
    final_out_shape = (None, hh, ww, out_shape)

    qkv = keras.layers.Dense(emb_dim * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], 3, num_heads, key_dim])
    query, key, value = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, batch, num_heads, blocks, key_dim]

    # query = key = [batch, num_heads, hh * ww, key_dim]
    query *= qk_scale
    # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True))([query, key])
    # pos_query = [batch, num_heads, hh, ww, key_dim]
    pos_query = tf.reshape(query, [-1, num_heads, inputs.shape[1], inputs.shape[2], key_dim])
    pos_emb = RelativePositionalEmbedding(use_absolute_pos=not relative, name=name and name + "pos_emb")(pos_query)
    pos_emb = tf.reshape(pos_emb, [-1, *attention_scores.shape[1:]])
    attention_scores = keras.layers.Add()([attention_scores, pos_emb])
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, hh * ww, key_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output.set_shape(final_out_shape)
    return attention_output


def bot_block(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    relative_pe=True,
    strides=1,
    target_dimension=2048,
    name="all2all",
    use_MHSA=True,
):
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        # padding = "SAME" if strides == 1 else "VALID"
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, name=name + "_shorcut_")
        bn_act = activation if use_MHSA else None
        # bn_act = None
        shortcut = batchnorm_with_activation(shortcut, activation=bn_act, zero_gamma=False, name=name + "_shorcut_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor

    nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=1, padding="VALID", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")

    if use_MHSA:  # BotNet block
        nn = mhsa_with_relative_position_embedding(nn, num_heads=heads, relative=relative_pe, out_weight=False, name=name + "_2_mhsa_")
        if strides != 1:
            # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
            # nn = keras.layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
            nn = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(nn)
    else:  # ResNet block
        nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=strides, padding="SAME", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_2_")

    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, padding="VALID", name=name + "_3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")

    nn = keras.layers.Add(name=name + "_add")([shortcut, nn])
    return keras.layers.Activation(activation, name=name + "_out")(nn)


def bot_stack(
    featuremap,
    target_dimension=2048,
    num_layers=3,
    strides=2,
    activation="relu",
    heads=4,
    proj_factor=4,
    relative_pe=True,
    name="all2all_stack",
    use_MHSA=True,
):
    """ c5 Blockgroup of BoT Blocks. Use `activation=swish` for `silu` """
    for i in range(num_layers):
        featuremap = bot_block(
            featuremap,
            heads=heads,
            proj_factor=proj_factor,
            activation=activation,
            relative_pe=relative_pe,
            strides=strides if i == 0 else 1,
            target_dimension=target_dimension,
            name=name + "block{}".format(i + 1),
            use_MHSA=use_MHSA,
        )
    return featuremap


def BotNet(
    num_blocks,
    strides=1,
    preact=False,
    use_bias=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="botnet",
    **kwargs
):
    inputs = keras.layers.Input(shape=input_shape)

    nn = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(inputs)
    nn = keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(nn)

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="conv1_")
    nn = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(nn)
    nn = keras.layers.MaxPooling2D(3, strides=2, name="pool1_pool")(nn)

    out_channels = [64, 128, 256, 512]
    stack_strides = [1, 2, 2, strides]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, stack_strides)):
        name = "stack{}_".format(id + 1)
        use_MHSA = True if id == len(num_blocks) - 1 else False  # use MHSA in the last stack
        nn = bot_stack(nn, out_channel * 4, num_block, strides=stride, activation=activation, relative_pe=True, name=name, use_MHSA=use_MHSA)

    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")
    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    reload_model_weights_with_mismatch(model, PRETRAINED_DICT, "botnet", RelativePositionalEmbedding, input_shape=input_shape, pretrained=pretrained)
    return model


def BotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    return BotNet(**locals(), model_name="botnet50", **kwargs)


def BotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    return BotNet(**locals(), model_name="botnet101", **kwargs)


def BotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    return BotNet(**locals(), model_name="botnet152", **kwargs)
