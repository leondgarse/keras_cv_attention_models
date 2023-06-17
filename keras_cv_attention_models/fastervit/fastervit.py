import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    conv2d_no_bias,
    batchnorm_with_activation,
    add_with_layer_scale_and_drop_block,
    pad_to_divisible_by_window_size,
    reverse_padded_for_window_size,
    window_partition,
    window_reverse,
    multi_head_self_attention,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_EPSILON = 1e-4
LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "": {"": {192: ""}},
}

# [TODO] deploy
class MlpPairwisePositionalEmbedding(layers.Layer):
    def __init__(self, hidden_dim=512, attn_height=-1, pos_scale=-1, use_absolute_pos=False, **kwargs):
        # No weight, just need to wrapper a layer, or will not in model structure
        super().__init__(**kwargs)
        self.hidden_dim, self.attn_height, self.pos_scale, self.use_absolute_pos = hidden_dim, attn_height, pos_scale, use_absolute_pos

    def _build_absolute_index_(self):
        hh, ww = np.meshgrid(range(0, self.height), range(0, self.width), indexing="ij")
        coords = np.stack([hh, ww], axis=-1).astype("float32")
        coords = coords / [self.height // 2, self.width // 2] - 1
        # coords = np.reshape(coords, [-1, 2])
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("coords", functional.convert_to_tensor(coords, dtype=self.compute_dtype), persistent=False)
        else:
            self.coords = functional.convert_to_tensor(coords, dtype=self.compute_dtype)

    def _build_relative_index_(self):
        hh, ww = np.meshgrid(range(self.height), range(self.width))
        coords = np.stack([hh, ww], axis=-1).astype("float32")  # [15, 12, 2]
        coords_flatten = np.reshape(coords, [-1, 2])  # [180, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [180, 180, 2]
        # relative_coords = tf.reshape(relative_coords, [-1, 2])  # [196 * 196, 2]

        relative_coords_hh = relative_coords[:, :, 0] + self.height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + self.width - 1) * (2 * self.height - 1)
        relative_coords_hhww = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)
        relative_position_index = np.sum(relative_coords_hhww, axis=-1)  # [180, 180]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("relative_position_index", functional.convert_to_tensor(relative_position_index, dtype="int64"), persistent=False)
        else:
            self.relative_position_index = functional.convert_to_tensor(relative_position_index, dtype="int64")

    def _build_relative_coords_(self):
        hh, ww = np.meshgrid(range(-self.height + 1, self.height), range(-self.width + 1, self.width), indexing="ij")
        coords = np.stack([hh, ww], axis=-1).astype("float32")
        if self.pos_scale == -1:
            pos_scale = [self.height, self.width]
        else:
            # If pretrined weights are from different input_shape or window_size, pos_scale is previous actually using window_size
            pos_scale = self.pos_scale if isinstance(self.pos_scale, (list, tuple)) else [self.pos_scale, self.pos_scale]
        coords = coords * 8 / [float(pos_scale[0] - 1), float(pos_scale[1] - 1)]  # [23, 29, 2], normalize to -8, 8
        # torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        coords = np.sign(coords) * np.log(1.0 + np.abs(coords)) / (np.log(2.0) * 3.0)
        coords = np.reshape(coords, [-1, 2])  # [23 * 29, 2]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("coords", functional.convert_to_tensor(coords, dtype=self.compute_dtype), persistent=False)
        else:
            self.coords = functional.convert_to_tensor(coords, dtype=self.compute_dtype)

    def build(self, input_shape):
        if self.use_absolute_pos:
            self.height, self.width = input_shape[1], input_shape[2]
            self._build_absolute_index_()
            out_shape = [self.hidden_dim, input_shape[-1]]
        else:
            # input_shape: [batch, num_heads, hh * ww, hh * ww]
            if self.attn_height == -1:
                height = width = int(float(input_shape[-2]) ** 0.5)  # hh == ww, e.g. 14
            else:
                height = self.attn_height
                width = input_shape[-2] // height
            self.height, self.width, self.num_heads = height, width, input_shape[1]

            self._build_relative_coords_()
            self._build_relative_index_()
            out_shape = [self.hidden_dim, self.num_heads]

        self.hidden_weight = self.add_weight(name="hidden_weight", shape=[2, self.hidden_dim], trainable=True)
        self.hidden_bias = self.add_weight(name="hidden_bias", shape=[self.hidden_dim], initializer="zeros", trainable=True)
        self.out = self.add_weight(name="out", shape=out_shape, trainable=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pos_bias = self.coords @ self.hidden_weight + self.hidden_bias
        pos_bias = functional.relu(pos_bias)
        pos_bias = pos_bias @ self.out

        if not self.use_absolute_pos:
            pos_bias = functional.gather(pos_bias, self.relative_position_index)  # [hh * ww, hh * ww, num_heads]
            pos_bias = functional.sigmoid(pos_bias) * 16.0
            pos_bias = functional.expand_dims(functional.transpose(pos_bias, [2, 0, 1]), 0)
        return inputs + pos_bias

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {"hidden_dim": self.hidden_dim, "attn_height": self.attn_height, "pos_scale": self.pos_scale, "use_absolute_pos": self.use_absolute_pos}
        )
        return base_config


def attention_mlp_block(inputs, num_heads=4, mlp_ratio=4, sr_ratio=1, pos_scale=-1, pos_hidden_dim=512, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    # channnel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[-1]
    pre = MlpPairwisePositionalEmbedding(hidden_dim=pos_hidden_dim, pos_scale=pos_scale, use_absolute_pos=True, name=name + "pos_")(inputs)

    if sr_ratio > 1:
        pass  # [TODO]

    """ attention """
    nn = layer_norm(pre, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "attn_")
    pos_emb = MlpPairwisePositionalEmbedding(hidden_dim=pos_hidden_dim, attn_height=inputs.shape[1], pos_scale=pos_scale, name=name + "hat_pos_")
    attn = multi_head_self_attention(nn, num_heads=num_heads, pos_emb=pos_emb, qkv_bias=True, out_bias=True, name=name + "hat_")
    attn_out = add_with_layer_scale_and_drop_block(pre, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "attn_")

    if sr_ratio > 1:
        pass  # [TODO]

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, use_bias=False, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "mlp_")
    return nn


def res_conv_bn_block(inputs, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]

    nn = conv2d_no_bias(inputs, input_channel, kernel_size=3, use_bias=True, padding="SAME", name=name + "1_")  # epsilon=1e-5
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = conv2d_no_bias(nn, input_channel, kernel_size=3, use_bias=True, padding="SAME", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")  # epsilon=1e-5
    nn = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name)
    return nn


def FastViT(
    num_blocks=[2, 3, 6, 5],
    num_heads=[2, 4, 8, 16],
    window_sizes=[8, 8, 7, 7],
    # block_types=["conv", "conv", "transform", "transform"],
    stem_hidden_dim=64,
    embed_dim=64,
    mlp_ratio=4,
    ct_size=2,
    pos_scale=-1,  # If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    layer_scale=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="fastervit",
    kwargs=None,
):
    """Patch stem"""
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, stem_hidden_dim, 3, strides=2, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation="relu", name="stem_1_")
    nn = conv2d_no_bias(nn, embed_dim, 3, strides=2, padding="same", name="stem_2_")
    nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation="relu", name="stem_2_")

    block_types = ["conv", "conv", "transform", "transform"]

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, block_type, window_size, num_head) in enumerate(zip(num_blocks, block_types, window_sizes, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channels = embed_dim * (2**stack_id)
        is_conv_block = True if block_type[0].lower() == "c" else False
        nn = nn if is_conv_block or backend.image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last

        if stack_id > 0:
            layer_norm_axis = -1 if image_data_format() == "channels_last" or not is_conv_block else 1
            nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, axis=layer_norm_axis, name=stack_name + "downsample_")
            nn = nn if is_conv_block or image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=stack_name + "permute_pre")(nn)
            nn = conv2d_no_bias(nn, out_channels, 3, strides=2, padding="same", name=stack_name + "downsample_")
            nn = nn if is_conv_block or image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=stack_name + "permute_post")(nn)

        if not is_conv_block:
            nn, window_height, window_width, padding_height, padding_width = pad_to_divisible_by_window_size(nn, window_size)
            patch_height, patch_width = nn.shape[1] // window_height, nn.shape[2] // window_width
            nn = window_partition(nn, window_height, window_width)
            sr_ratio = max(patch_height, patch_width)

        for block_id in range(num_block):
            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block:
                nn = res_conv_bn_block(nn, layer_scale=layer_scale, drop_rate=block_drop_rate, activation=activation, name=name)
            else:
                nn = attention_mlp_block(
                    nn, num_head, mlp_ratio, sr_ratio, pos_scale=pos_scale, layer_scale=layer_scale, drop_rate=block_drop_rate, activation=activation, name=name
                )

        if not is_conv_block:
            nn = window_reverse(nn, patch_height, patch_width)
            nn = reverse_padded_for_window_size(nn, padding_height, padding_width)
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name="permute_post")(nn)

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "fastervit", pretrained)
    return model


def FastViT0(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return FastViT(**locals(), model_name="fastervit_0", **kwargs)
