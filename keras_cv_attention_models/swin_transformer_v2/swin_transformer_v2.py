import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    drop_block,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "swin_transformer_v2_tiny_ns": {"imagenet": {224: "c3272af88ba0cf09c818ac558ca9970e"}},
    "swin_transformer_v2_small": {"imagenet": {224: "d885a15b6d19cf72eee3a43d9c548579"}},
}


@tf.keras.utils.register_keras_serializable(package="swinv2")
class DivideScale(keras.layers.Layer):
    def __init__(self, axis=-1, initializer="ones", min_value=0.01, **kwargs):
        super().__init__(**kwargs)
        self.axis, self.initializer, self.min_value = axis, initializer, min_value

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            weight_shape = (input_shape[-1],)
        else:
            weight_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                weight_shape[ii] = input_shape[ii]
        self.scale = self.add_weight(name="weight", shape=weight_shape, initializer=self.initializer, trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs / tf.maximum(self.scale, self.min_value)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "min_value": self.min_value})  # Not saving initializer in config
        return config


@tf.keras.utils.register_keras_serializable(package="swinv2")
class PairWiseRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [batch * window_patch, window_height, window_width, channel]
        height, width = input_shape[1], input_shape[2]
        xx, yy = tf.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = tf.stack([yy, xx], axis=-1)  # [14, 14, 2]
        coords_flatten = tf.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        # relative_coords = tf.reshape(relative_coords, [-1, 2])  # [196 * 196, 2]
        relative_coords = tf.cast(relative_coords, self.dtype)
        self.relative_coords_log = tf.sign(relative_coords) * tf.math.log(1.0 + tf.abs(relative_coords))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.relative_coords_log


def window_multi_head_self_attention(inputs, filters=-1, num_heads=4, meta_hidden_dim=384, mask=None, out_bias=True, attn_dropout=0, out_dropout=0, name=None):
    input_channel = inputs.shape[-1]
    filters = filters if filters > 0 else input_channel
    key_dim = input_channel // num_heads

    qkv = keras.layers.Dense(input_channel * 3, use_bias=True, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, 3, axis=-1)
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    norm_query, norm_key = tf.norm(query, axis=-1, keepdims=True), tf.norm(key, axis=-2, keepdims=True)
    attn = tf.matmul(query, key) / tf.maximum(tf.matmul(norm_query, norm_key), 1e-6)
    attn = DivideScale(axis=1, name=name and name + "scale")(attn)  # On head dim

    # _relative_positional_encodings
    pos_coord = PairWiseRelativePositionalEmbedding(name=name and name + "pos_emb")(inputs) # Wrapper a layer, or will not in model structure
    relative_position_bias = mlp_block(pos_coord, meta_hidden_dim, output_channel=num_heads, drop_rate=0.1, activation="relu", name=name and name + "meta_")
    relative_position_bias = tf.expand_dims(tf.transpose(relative_position_bias, [2, 0, 1]), 0)
    attn = attn + relative_position_bias

    if mask is not None:
        query_blocks = attn.shape[2]
        attn = tf.reshape(attn, [-1, mask.shape[0], num_heads, query_blocks, query_blocks])
        attn += tf.expand_dims(tf.expand_dims(mask, 1), 0)  # expand dims on batch and num_heads
        attn = tf.reshape(attn, [-1, num_heads, query_blocks, query_blocks])
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    attention_output = tf.matmul(attention_scores, value)
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
    attention_output = keras.layers.Dense(input_channel, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(out_dropout, name=name and name + "out_drop")(attention_output) if out_dropout > 0 else attention_output
    return attention_output


def make_window_attention_mask(height, width, window_height, window_width, shift_height, shift_width):
    float_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    # TODO: use stack instead of assign value
    hh_split = [0, -window_height, -shift_height, None]
    ww_split = [0, -window_width, -shift_width, None]
    mask = tf.zeros([height, width]).numpy()  # need to assign values
    mask_value = 0  # value is ignored
    for hh_start, hh_end in zip(hh_split[:-1], hh_split[1:]):
        for ww_start, ww_end in zip(ww_split[:-1], ww_split[1:]):
            mask[hh_start:hh_end, ww_start:ww_end] = mask_value
            mask_value += 1
    mask = tf.convert_to_tensor(mask)
    # return mask

    mask = tf.reshape(mask, [height // window_height, window_height, width // window_width, window_width])
    mask = tf.transpose(mask, [0, 2, 1, 3])
    mask = tf.reshape(mask, [-1, window_height * window_width])
    attn_mask = tf.expand_dims(mask, 1) - tf.expand_dims(mask, 2)
    return tf.cast(tf.where(attn_mask != 0, -100, 0), float_dtype)


def shifted_window_attention(inputs, window_size, num_heads=4, shift_size=0, name=""):
    input_channel = inputs.shape[-1]
    window_height = window_size[0] if window_size[0] < inputs.shape[1] else inputs.shape[1]
    window_width = window_size[1] if window_size[1] < inputs.shape[2] else inputs.shape[2]
    shift_size = 0 if (window_height == inputs.shape[1] and window_width == inputs.shape[2]) else shift_size
    should_shift = shift_size > 0

    if should_shift:
        shift_height, shift_width = int(window_height * shift_size), int(window_width * shift_size)
        inputs = tf.roll(inputs, shift=(shift_height * -1, shift_width * -1), axis=[1, 2])
        mask = make_window_attention_mask(inputs.shape[1], inputs.shape[2], window_height, window_width, shift_height, shift_width)
    else:
        mask = None

    # window_partition, partition windows
    patch_height, patch_width = inputs.shape[1] // window_height, inputs.shape[2] // window_width
    nn = tf.reshape(inputs, [-1, patch_height, window_height, patch_width, window_width, input_channel])
    nn = tf.transpose(nn, [0, 1, 3, 2, 4, 5])
    nn = tf.reshape(nn, [-1, window_height, window_width, input_channel])

    nn = window_multi_head_self_attention(nn, num_heads=num_heads, mask=mask, name=name)

    # window_reverse, merge windows
    nn = tf.reshape(nn, [-1, patch_height, patch_width, window_height, window_width, input_channel])
    nn = tf.transpose(nn, [0, 1, 3, 2, 4, 5])
    nn = tf.reshape(nn, [-1, patch_height * window_height, patch_width * window_width, input_channel])

    if should_shift:
        nn = tf.roll(nn, shift=(shift_height, shift_width), axis=[1, 2])
    return nn


def swin_transformer_block(
    inputs, window_size, num_heads=4, shift_size=0, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, layer_scale=-1, name=None
):
    input_channel = inputs.shape[-1]
    attn = shifted_window_attention(inputs, window_size, num_heads, shift_size, name=name + "attn_")
    attn = layer_norm(attn, name=name + "attn_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = mlp_block(attn_out, int(input_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=False, activation="gelu", name=name + "mlp_")
    mlp = layer_norm(mlp, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


def patch_merging(inputs, name=""):
    input_channel = inputs.shape[-1]
    nn = tf.reshape(inputs, [-1, inputs.shape[1] // 2, 2, inputs.shape[2] // 2, 2, input_channel])
    nn = tf.transpose(nn, [0, 1, 3, 4, 2, 5])
    nn = tf.reshape(nn, [-1, nn.shape[1], nn.shape[2], 2 * 2 * input_channel])
    nn = layer_norm(nn, name=name)
    nn = keras.layers.Dense(2 * input_channel, use_bias=False, name=name + "dense")(nn)
    return nn


def SwinTransformerV2(
    num_blocks=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    embed_dim=96,
    window_ratio=32,
    stem_patch_size=4,
    use_stack_norm=False,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="swin_transformer_v2",
    kwargs=None,
):
    """ Patch stem """
    inputs = keras.layers.Input(input_shape)
    nn = keras.layers.Conv2D(embed_dim, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, name="stem_conv")(inputs)
    nn = layer_norm(nn, name="stem_")
    window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio]

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    # drop_connect_rates = drop_connect_rates_split(num_blocks, start=0.0, end=drop_connect_rate)
    # embed_dim = stem_width
    for id, (num_block, num_head) in enumerate(zip(num_blocks, num_heads)):
        stack_name = "stack{}_".format(id + 1)
        if id > 0:
            nn = patch_merging(nn, name=stack_name + "downsample")
            # embed_dim *= 2
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            shift_size = 0 if block_id % 2 == 0 else 0.5
            nn = swin_transformer_block(nn, window_size, num_head, shift_size, drop_rate=block_drop_rate, layer_scale=layer_scale, name=block_name)
            global_block_id += 1
        if use_stack_norm and id != len(num_blocks) - 1:  # Exclude last stack
            nn = layer_norm(nn, name=stack_name + "output_")
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "swin_transformer_v2", pretrained)
    return model


def SwinTransformerV2Tiny(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny", **kwargs)


def SwinTransformerV2Tiny_ns(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_stack_norm = True
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny_ns", **kwargs)


def SwinTransformerV2Small(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_small", **kwargs)


def SwinTransformerV2Base(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_base", **kwargs)


def SwinTransformerV2Large(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]
    embed_dim = 192
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_large", **kwargs)
