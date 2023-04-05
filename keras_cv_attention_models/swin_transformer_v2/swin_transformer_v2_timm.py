import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers
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
    "swin_transformer_v2_small_ns": {"imagenet": {224: "89d5a63d528bbb88a4a287871e868414"}},
}


@backend.register_keras_serializable(package="swinv2")
class DivideScale(layers.Layer):
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
        self.scale = self.add_weight(name="weight", shape=weight_shape, initializer=self.initializer, trainable=True, dtype="float32")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs / functional.maximum(self.scale, self.min_value)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "min_value": self.min_value})  # Not saving initializer in config
        return config


@backend.register_keras_serializable(package="swinv2")
class PairWiseRelativePositionalEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        # No weight, just need to wrapper a layer, or will not in model structure
        super().__init__(**kwargs)
        self.use_layer_as_module = True

    def build(self, input_shape):
        # input_shape: [batch * window_patch, window_height, window_width, channel]
        height, width = input_shape[1], input_shape[2]
        xx, yy = np.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = np.stack([yy, xx], axis=-1).astype("float32")  # [14, 14, 2]
        coords_flatten = np.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        # relative_coords = tf.reshape(relative_coords, [-1, 2])  # [196 * 196, 2]
        # relative_coords = tf.cast(relative_coords, self.dtype)
        relative_coords_log = np.sign(relative_coords) * np.log(1.0 + np.abs(relative_coords))

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("relative_coords_log", functional.convert_to_tensor(relative_coords_log, dtype="float32"), persistent=False)
        else:
            self.relative_coords_log = functional.convert_to_tensor(relative_coords_log, dtype="float32")
        self.height, self.width = height, width
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.relative_coords_log

    def compute_output_shape(self, input_shape):
        return [self.height * self.width, self.height * self.width, 2]


@backend.register_keras_serializable(package="swinv2")
class WindowAttentionMask(layers.Layer):
    def __init__(self, height, width, window_height, window_width, shift_height=0, shift_width=0, **kwargs):
        # No weight, just need to wrapper a layer, or will meet some error in model saving or loading...
        self.height, self.width, self.window_height, self.window_width = height, width, window_height, window_width
        self.shift_height, self.shift_width = shift_height, shift_width
        self.blocks = (self.height // self.window_height) * (self.width // self.window_width)
        super().__init__(**kwargs)

    def build(self, input_shape):
        hh_split = [0, self.height - self.window_height, self.height - self.shift_height, self.height]
        ww_split = [0, self.width - self.window_width, self.width - self.shift_width, self.width]
        mask_value, total_ww, mask = 0, len(ww_split) - 1, []
        for hh_id in range(len(hh_split) - 1):
            hh = hh_split[hh_id + 1] - hh_split[hh_id]
            rr = [np.zeros([hh, ww_split[id + 1] - ww_split[id]], dtype="float32") + (id + mask_value) for id in range(total_ww)]
            mask.append(np.concatenate(rr, axis=-1))
            mask_value += total_ww
        mask = np.concatenate(mask, axis=0)
        # return mask

        mask = np.reshape(mask, [self.height // self.window_height, self.window_height, self.width // self.window_width, self.window_width])
        mask = np.transpose(mask, [0, 2, 1, 3])
        mask = np.reshape(mask, [-1, self.window_height * self.window_width])
        attn_mask = np.expand_dims(mask, 1) - np.expand_dims(mask, 2)
        # attn_mask = tf.cast(np.where(attn_mask != 0, -100, 0), self._compute_dtype)
        attn_mask = np.where(attn_mask != 0, -100, 0)
        attn_mask = np.expand_dims(np.expand_dims(attn_mask, 1), 0)  # expand dims on batch and num_heads
        # attn_mask = functional.convert_to_tensor(attn_mask, dtype="float32")

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("attn_mask", functional.convert_to_tensor(attn_mask, dtype="float32"), persistent=False)
        else:
            self.attn_mask = functional.convert_to_tensor(attn_mask, dtype="float32")

        self.num_heads, self.query_blocks = input_shape[1], input_shape[2]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: [batch_size * blocks, num_heads, query_blocks, query_blocks]
        # where query_blocks = `window_height * window_width`, blocks = `(height // window_height) * (width // window_width)`
        nn = functional.reshape(inputs, [-1, self.blocks, self.num_heads, self.query_blocks, self.query_blocks])
        nn = nn + self.attn_mask
        return functional.reshape(nn, [-1, self.num_heads, self.query_blocks, self.query_blocks])

    def compute_output_shape(self, input_shape):
        return [None, self.num_heads, self.query_blocks, self.query_blocks]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "window_height": self.window_height,
                "window_width": self.window_width,
                "shift_height": self.shift_height,
                "shift_width": self.shift_width,
            }
        )
        return config


def window_mhsa_with_pair_wise_positional_embedding(
    inputs, num_heads=4, key_dim=0, meta_hidden_dim=384, mask=None, out_bias=True, attn_dropout=0, out_dropout=0, name=None
):
    input_channel = inputs.shape[-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    qk_out = key_dim * num_heads

    qkv = layers.Dense(qk_out * 3, use_bias=True, name=name and name + "qkv")(inputs)
    qkv = functional.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = functional.split(qkv, 3, axis=-1)
    query = functional.transpose(functional.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]
    key = functional.transpose(functional.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = functional.transpose(functional.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    norm_query, norm_key = functional.l2_normalize(query, axis=-1, epsilon=1e-6), functional.l2_normalize(key, axis=-2, epsilon=1e-6)
    attn = functional.matmul(norm_query, norm_key)
    attn = DivideScale(axis=1, name=name and name + "scale")(attn)  # axis=1 means on head dimension

    # _relative_positional_encodings
    pos_coord = PairWiseRelativePositionalEmbedding(name=name and name + "pos_emb")(inputs)  # Wrapper a layer, or will not in model structure
    relative_position_bias = mlp_block(pos_coord, meta_hidden_dim, output_channel=num_heads, drop_rate=0.1, activation="relu", name=name and name + "meta_")
    relative_position_bias = functional.expand_dims(functional.transpose(relative_position_bias, [2, 0, 1]), 0)
    attn = attn + relative_position_bias

    if mask is not None:
        attn = mask(attn)
    attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if attn_dropout > 0:
        attention_scores = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    attention_output = functional.matmul(attention_scores, value)
    attention_output = functional.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = functional.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
    attention_output = layers.Dense(qk_out, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = layers.Dropout(out_dropout, name=name and name + "out_drop")(attention_output) if out_dropout > 0 else attention_output
    return attention_output


def shifted_window_attention(inputs, window_size, num_heads=4, shift_size=0, name=None):
    input_channel = inputs.shape[-1]
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    window_height = window_size[0] if window_size[0] < inputs.shape[1] else inputs.shape[1]
    window_width = window_size[1] if window_size[1] < inputs.shape[2] else inputs.shape[2]
    shift_size = 0 if (window_height == inputs.shape[1] and window_width == inputs.shape[2]) else shift_size
    should_shift = shift_size > 0

    # window_partition, partition windows, ceil mode padding if not divisible by window_size
    # patch_height, patch_width = inputs.shape[1] // window_height, inputs.shape[2] // window_width
    patch_height, patch_width = int(math.ceil(inputs.shape[1] / window_height)), int(math.ceil(inputs.shape[2] / window_width))
    should_pad_hh, should_pad_ww = patch_height * window_height - inputs.shape[1], patch_width * window_width - inputs.shape[2]
    # print(f">>>> shifted_window_attention {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = functional.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    if should_shift:
        shift_height, shift_width = int(window_height * shift_size), int(window_width * shift_size)
        # tf.roll is not supported by tflite
        # inputs = tf.roll(inputs, shift=(shift_height * -1, shift_width * -1), axis=[1, 2])
        inputs = functional.concat([inputs[:, shift_height:], inputs[:, :shift_height]], axis=1)
        inputs = functional.concat([inputs[:, :, shift_width:], inputs[:, :, :shift_width]], axis=2)

    # print(f">>>> shifted_window_attention {inputs.shape = }, {patch_height = }, {patch_width = }, {window_height = }, {window_width = }")
    # [batch * patch_height, window_height, patch_width, window_width * channel], limit transpose perm <= 4
    nn = functional.reshape(inputs, [-1, window_height, patch_width, window_width * input_channel])
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, patch_width, window_height, window_width * channel]
    nn = functional.reshape(nn, [-1, window_height, window_width, input_channel])  # [batch * patch_height * patch_width, window_height, window_width, channel]

    mask = WindowAttentionMask(inputs.shape[1], inputs.shape[2], window_height, window_width, shift_height, shift_width) if should_shift else None
    nn = window_mhsa_with_pair_wise_positional_embedding(nn, num_heads=num_heads, mask=mask, name=name)

    # window_reverse, merge windows
    # [batch * patch_height, patch_width, window_height, window_width * input_channel], limit transpose perm <= 4
    nn = functional.reshape(nn, [-1, patch_width, window_height, window_width * input_channel])
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, window_height, patch_width, window_width * input_channel]
    nn = functional.reshape(nn, [-1, patch_height * window_height, patch_width * window_width, input_channel])

    if should_shift:
        # nn = tf.roll(nn, shift=(shift_height, shift_width), axis=[1, 2])
        nn = functional.concat([nn[:, -shift_height:], nn[:, :-shift_height]], axis=1)
        nn = functional.concat([nn[:, :, -shift_width:], nn[:, :, :-shift_width]], axis=2)

    # print(f">>>> shifted_window_attention before: {nn.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        nn = nn[:, : nn.shape[1] - should_pad_hh, : nn.shape[2] - should_pad_ww, :]  # In case should_pad_hh or should_pad_ww is 0
    # print(f">>>> shifted_window_attention after: {nn.shape = }")

    return nn


def swin_transformer_block(
    inputs, window_size, num_heads=4, shift_size=0, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, layer_scale=-1, name=None
):
    input_channel = inputs.shape[-1]
    attn = shifted_window_attention(inputs, window_size, num_heads, shift_size, name=name + "attn_")
    attn = layer_norm(attn, zero_gamma=True, axis=-1, name=name + "attn_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = mlp_block(attn_out, int(input_channel * mlp_ratio), drop_rate=mlp_drop_rate, activation="gelu", name=name + "mlp_")
    mlp = layer_norm(mlp, zero_gamma=True, axis=-1, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return layers.Add(name=name + "output")([attn_out, mlp])


def patch_merging(inputs, name=""):
    input_channel = inputs.shape[-1]
    should_pad_hh, should_pad_ww = inputs.shape[1] % 2, inputs.shape[2] % 2
    # print(f">>>> patch_merging {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = functional.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    # limit transpose perm <= 4
    nn = functional.reshape(inputs, [-1, 2, inputs.shape[2], input_channel])  # [batch * inputs.shape[1] // 2, height 2, inputs.shape[2], input_channel]
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * inputs.shape[1] // 2, inputs.shape[2], height 2, input_channel]
    nn = functional.reshape(nn, [-1, inputs.shape[1] // 2, inputs.shape[2] // 2, 2 * 2 * input_channel])
    nn = layer_norm(nn, axis=-1, name=name)
    nn = layers.Dense(2 * input_channel, use_bias=False, name=name + "dense")(nn)
    return nn


def SwinTransformerV2(
    num_blocks=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    embed_dim=96,
    # window_ratio=32,
    window_size=7,
    stem_patch_size=4,
    use_stack_norm=False,  # True for extra layer_norm on each stack end
    extra_norm_period=0,  # > 0 for extra layer_norm frequency in each stack. May combine with use_stack_norm=True
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
    """Patch stem"""
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = layers.Conv2D(embed_dim, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, name="stem_conv")(inputs)
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
    nn = layer_norm(nn, axis=-1, name="stem_")
    # window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio]
    # window_size = [int(tf.math.ceil(input_shape[0] / window_ratio)), int(tf.math.ceil(input_shape[1] / window_ratio))]
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, num_head) in enumerate(zip(num_blocks, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            # height, width downsample * 0.5, channel upsample * 2
            nn = patch_merging(nn, name=stack_name + "downsample")
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            shift_size = 0 if block_id % 2 == 0 else 0.5
            nn = swin_transformer_block(nn, window_size, num_head, shift_size, drop_rate=block_drop_rate, layer_scale=layer_scale, name=block_name)
            global_block_id += 1
            if extra_norm_period > 0 and (block_id + 1) % extra_norm_period == 0 and not (use_stack_norm and block_id == num_block - 1):
                nn = layer_norm(nn, axis=-1, name=block_name + "output_")
        if use_stack_norm and stack_id != len(num_blocks) - 1:  # Exclude last stack
            nn = layer_norm(nn, axis=-1, name=stack_name + "output_")
    nn = layer_norm(nn, axis=-1, name="pre_output_")
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "swin_transformer_v2", pretrained)
    return model


def SwinTransformerV2Tiny_ns(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_stack_norm = True
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny_ns", **kwargs)


def SwinTransformerV2Small_ns(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    use_stack_norm = True
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_small_ns", **kwargs)
