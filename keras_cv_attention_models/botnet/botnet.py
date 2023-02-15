import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, initializers, image_data_format
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.attention_layers import conv2d_no_bias, scaled_dot_product_attention, qkv_to_multi_head_channels_last_format
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

PRETRAINED_DICT = {
    # "botnet50": {"imagenet": "b221b45ca316166fc858fda1cf4fd946"},
    "botnet26t": {"imagenet": {256: "6d7a9548f866b4971ca2c9d17dd815fc"}},
    "botnext_eca26t": {"imagenet": {256: "170b9b4d7fba88dbcb41716047c047b9"}},
    "botnet_se33t": {"imagenet": {256: "f612743ec59d430f197bc38b3a7f8837"}},
}


@backend.register_keras_serializable(package="botnet")
class RelativePositionalEmbedding(layers.Layer):
    def __init__(self, position_height=0, position_width=0, use_absolute_pos=False, dynamic_shape=False, **kwargs):
        super().__init__(**kwargs)
        self.position_height = position_height
        self.position_width = position_width if position_width > 0 else position_height
        self.use_absolute_pos = use_absolute_pos
        self.dynamic_shape = dynamic_shape

    def build(self, input_shape):
        _, num_heads, height, width, key_dim = input_shape
        self.position_height = self.position_height if self.position_height > height else height
        self.position_width = self.position_width if self.position_width > width else width
        self.key_dim = key_dim
        stddev = key_dim**-0.5

        if self.use_absolute_pos:
            hh_shape = (key_dim, self.position_height)
            ww_shape = (key_dim, self.position_width)
        else:
            hh_shape = (key_dim, 2 * self.position_height - 1)
            ww_shape = (key_dim, 2 * self.position_width - 1)

        initializer = initializers.random_normal(stddev=stddev)
        self.pos_emb_h = self.add_weight(name="r_height", shape=hh_shape, initializer=initializer, trainable=True)
        self.pos_emb_w = self.add_weight(name="r_width", shape=ww_shape, initializer=initializer, trainable=True)
        self.input_height, self.input_width = height, width
        super().build(input_shape)

    def get_config(self):
        base_config = super(RelativePositionalEmbedding, self).get_config()
        base_config.update(
            {
                "position_height": self.position_height,
                "position_width": self.position_width,
                "use_absolute_pos": self.use_absolute_pos,
                "dynamic_shape": self.dynamic_shape,
            }
        )
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs+heads, height, width, 2 * pos_dim - 1]
        Output: [bs+heads, height, width, pos_dim]
        """
        bs_heads, hh, ww, dim = rel_pos.shape  # [bs+heads, height, width, 2 * width - 1]
        pos_dim = (dim + 1) // 2
        if pos_dim == 1:
            return rel_pos
        if ww == 1:
            return rel_pos[:, :, :, -pos_dim:]
        full_rank_gap = pos_dim - ww
        # [bs+heads, height, width * (2 * pos_dim - 1)] --> [bs+heads, height, width * (2 * pos_dim - 1) - width]
        flat_x = functional.reshape(rel_pos, [-1, hh, ww * dim])[:, :, ww - 1 : -1]
        # [bs+heads, height, width, 2 * (pos_dim - 1)] --> [bs+heads, height, width, pos_dim]
        # print(f">>>> {full_rank_gap = }, {flat_x.shape = }")
        return functional.reshape(flat_x, [-1, hh, ww, 2 * (pos_dim - 1)])[:, :, :, full_rank_gap : pos_dim + full_rank_gap]

    def relative_logits(self, inputs):
        bs, heads, hh, ww, cc = inputs.shape  # e.g.: [1, 4, 14, 16, 128]
        inputs = functional.reshape(inputs, [-1, hh, ww, cc])  # Merge bs and heads, for supporting TFLite conversion
        rel_logits_w = functional.matmul(inputs, self.pos_emb_w)  # [4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)  # [4, 14, 16, 16]

        query_h = functional.transpose(inputs, [0, 2, 1, 3])  # [4, 16, 14, 128], [bs+heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = functional.matmul(query_h, self.pos_emb_h)  # [4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [4, 16, 14, 14]
        rel_logits_h = functional.transpose(rel_logits_h, [0, 2, 1, 3])  # [4, 14, 16, 14], transpose back

        logits = functional.expand_dims(rel_logits_w, axis=-2) + functional.expand_dims(rel_logits_h, axis=-1)  # [4, 14, 16, 14, 16]
        return functional.reshape(logits, [-1, heads, hh, ww, self.position_height, self.position_width])  # [1, 4, 14, 16, 14, 16]

    def absolute_logits(self, inputs):
        # pos_emb = tf.expand_dims(self.pos_emb_w, -2) + tf.expand_dims(self.pos_emb_h, -1)
        # return tf.einsum("bxyhd,dpq->bhxypq", inputs, pos_emb)
        rel_logits_w = functional.matmul(inputs, self.pos_emb_w)
        rel_logits_h = functional.matmul(inputs, self.pos_emb_h)
        return functional.expand_dims(rel_logits_w, axis=-2) + functional.expand_dims(rel_logits_h, axis=-1)

    def call(self, inputs):
        pos_emb = self.absolute_logits(inputs) if self.use_absolute_pos else self.relative_logits(inputs)
        if self.dynamic_shape:
            _, _, hh, ww, _ = inputs.shape
            if hh < self.position_height or ww < self.position_width:
                pos_emb = pos_emb[:, :, :, :, :hh, :ww]
        return pos_emb

    def load_resized_weights(self, source_layer, method="bilinear"):
        # For input 224 --> [128, 27], convert to 480 --> [128, 30]
        if isinstance(source_layer, dict):
            source_pos_emb_h, source_pos_emb_w = list(source_layer.values())
        else:
            source_pos_emb_h, source_pos_emb_w = source_layer.pos_emb_h, source_layer.pos_emb_w  # layer
        source_pos_emb_h = np.array(source_pos_emb_h.detach() if hasattr(source_pos_emb_h, "detach") else source_pos_emb_h).astype("float32")
        source_pos_emb_w = np.array(source_pos_emb_w.detach() if hasattr(source_pos_emb_w, "detach") else source_pos_emb_w).astype("float32")

        image_like_h = np.expand_dims(np.transpose(source_pos_emb_h, [1, 0]), 0)
        resize_h = backend.numpy_image_resize(image_like_h, target_shape=(1, self.pos_emb_h.shape[1]), method=method)[0]
        resize_h = np.transpose(resize_h, [1, 0])

        image_like_w = np.expand_dims(np.transpose(source_pos_emb_w, [1, 0]), 0)
        resize_w = backend.numpy_image_resize(image_like_w, target_shape=(1, self.pos_emb_w.shape[1]), method=method)[0]
        resize_w = np.transpose(resize_w, [1, 0])

        self.set_weights([resize_h, resize_w])

    def show_pos_emb(self, base_size=4):
        import matplotlib.pyplot as plt

        pos_emb_h = self.pos_emb_h.detach().numpy() if hasattr(self.pos_emb_h, "detach") else self.pos_emb_h.numpy()
        pos_emb_w = self.pos_emb_w.detach().numpy() if hasattr(self.pos_emb_w, "detach") else self.pos_emb_w.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(base_size * 3, base_size * 1))
        axes[0].imshow(pos_emb_h)
        axes[1].imshow(pos_emb_w)
        hh_sum = np.ones([1, pos_emb_h.shape[0]]) @ pos_emb_h
        ww_sum = np.ones([1, pos_emb_w.shape[0]]) @ pos_emb_w
        axes[2].imshow(np.transpose(hh_sum) + ww_sum)
        titles = ["pos_emb_h", "pos_emb_w", "sum"]
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def mhsa_with_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, relative=True, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, data_format=None, name=None
):
    data_format = image_data_format() if data_format is None else data_format
    channel_axis = -1 if data_format == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    hh, ww = inputs.shape[1:-1] if data_format == "channels_last" else inputs.shape[2:]

    key_dim = key_dim if key_dim > 0 else input_channels // num_heads
    out_shape = input_channels if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads

    # qkv = layers.Dense(qk_out * 2 + out_shape, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = conv2d_no_bias(inputs, qk_out * 2 + out_shape, kernel_size=1, name=name and name + "qkv_")
    # qkv = functional.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], qkv.shape[-1]])
    query, key, value = functional.split(qkv, [qk_out, qk_out, out_shape], axis=channel_axis)
    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format=data_format)

    # pos_query = [batch, num_heads, hh, ww, key_dim]
    pos_query = functional.reshape(query, [-1, num_heads, hh, ww, key_dim])
    pos_emb = RelativePositionalEmbedding(use_absolute_pos=not relative, name=name and name + "pos_emb")(pos_query)

    output_shape = [hh, ww, out_shape]
    pos_emb_func = lambda attention_scores: attention_scores + functional.reshape(pos_emb, [-1, *attention_scores.shape[1:]])
    out = scaled_dot_product_attention(query, key, value, output_shape, pos_emb_func, out_weight, out_bias, dropout=attn_dropout, name=name)
    return out if data_format == "channels_last" else layers.Permute([3, 1, 2])(out)

    # query *= qk_scale
    # [batch, num_heads, hh * ww, hh * ww]
    # attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    # pos_emb = functional.reshape(pos_emb, [-1, *attention_scores.shape[1:]])
    # attention_scores = keras.layers.Add()([attention_scores, pos_emb])
    # # attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    # attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    #
    # if attn_dropout > 0:
    #     attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # # value = [batch, num_heads, hh * ww, vv_dim]
    # # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, vv_dim]
    # attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    # attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    # attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")
    #
    # if out_weight:
    #     # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
    #     attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    # return attention_output


def BotNet(input_shape=(224, 224, 3), strides=1, pretrained="imagenet", **kwargs):
    attn_types = [None, None, None, "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]

    model = AotNet(input_shape=input_shape, attn_types=attn_types, attn_params=attn_params, strides=strides, **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
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


def BotNet26T(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    stem_type = "tiered"

    model = AotNet(model_name="botnet26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model


def BotNextECA26T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "key_dim": 16, "out_weight": False}
    use_eca = True
    group_size = 16
    stem_type = "tiered"
    model = AotNet(model_name="botnext_eca26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model


def BotNetSE33T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_type = "tiered"
    stem_last_strides = 2
    stem_downsample = False
    out_channels = [256, 512, 1024, 1536]
    hidden_channel_ratio = [1 / 4, 1 / 4, 1 / 4, 1 / 3]
    num_blocks = [2, 3, 3, 2]
    attn_types = [None, [None, None, "bot"], [None, None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    se_ratio = 1 / 16
    output_num_features = 1280

    model = AotNet(model_name="botnet_se33t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model
