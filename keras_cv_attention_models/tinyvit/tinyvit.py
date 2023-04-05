import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    inverted_residual_block,
    layer_norm,
    mlp_block,
    mhsa_with_multi_head_position,
    window_attention,
    MultiHeadPositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "tiny_vit_11m": {"imagenet": {224: "6cc52dda567fd70f706d57c69dfd81d8"}, "imagenet21k-ft1k": {224: "f84673169e5c7a4ec526866da24cd5f4"}},
    "tiny_vit_21m": {
        "imagenet": {224: "5809a53bc3abe0785475c3f2d501a039"},
        "imagenet21k-ft1k": {224: "08d16dd06ddd85c2e4d2d15143c24607", 384: "fe6d364e99fa5a7f255ad3f3270bc962", 512: "ba5822042f0cb09bd8189290164b6ec3"},
    },
    "tiny_vit_5m": {"imagenet": {224: "a9c53f53e6da6a9b2edf6e773e5e402b"}, "imagenet21k-ft1k": {224: "cd10dbebf0645769dcda3224a0f330c4"}},
}


def tiny_vit_block(inputs, window_size=7, num_heads=4, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]

    """ attention """
    nn = inputs if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(inputs)  # channels_first -> channels_last
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "attn_")
    nn = window_attention(
        nn, window_size, num_heads=num_heads, attention_block=mhsa_with_multi_head_position, use_bn=False, qkv_bias=True, out_bias=True, name=name + "attn_"
    )
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last, channels_first
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "attn_")

    pre_mlp = depthwise_conv2d_no_bias(attn_out, kernel_size=3, strides=1, padding="SAME", name=name + "pre_mlp_")
    pre_mlp = batchnorm_with_activation(pre_mlp, activation=None, name=name + "pre_mlp_")

    """ MLP """
    nn = pre_mlp if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(pre_mlp)  # channels_first -> channels_last
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name + "mlp_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last, channels_first
    nn = add_with_layer_scale_and_drop_block(pre_mlp, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")
    return nn


def TinyViT(
    num_blocks=[2, 2, 6, 2],
    out_channels=[64, 128, 160, 320],
    block_types=["conv", "transform", "transform", "transform"],
    num_heads=[2, 4, 5, 10],
    # window_sizes=[7, 7, 14, 7],
    window_ratios=[8, 4, 1, 1],  # For `input_shape=(224, 224, 3)` will be window_sizes=[7, 7, 14, 7], for `(384, 384, 3)` will be `[12, 12, 24, 12]`.
    mlp_ratio=4,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    layer_scale=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="tiny_vit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, out_channels[0] // 2, kernel_size=3, strides=2, padding="SAME", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, out_channels[0], kernel_size=3, strides=2, padding="SAME", name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_2_")

    inverted_residual_block_kwargs = {
        "stride": 1,
        "expand": 4,
        "shortcut": True,
        "is_torch_mode": True,
        "use_last_bn_zero_gamma": True,
        "activation": activation,
    }

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            name = stack_name + "downsample_"
            expand = out_channel / nn.shape[-1 if image_data_format() == "channels_last" else 1]
            nn = inverted_residual_block(nn, out_channel, stride=2, expand=expand, shortcut=False, is_torch_mode=True, activation=activation, name=name)

        is_conv_block = True if block_type[0].lower() == "c" else False
        num_head = num_heads[stack_id] if isinstance(num_heads, (list, tuple)) else num_heads
        # window_size = window_sizes[stack_id] if isinstance(window_sizes, (list, tuple)) else window_sizes
        window_ratio = window_ratios[stack_id] if isinstance(window_ratios, (list, tuple)) else window_ratios
        height, width = nn.shape[1:-1] if image_data_format() == "channels_last" else nn.shape[2:]
        window_size = [int(math.ceil(height / window_ratio)), int(math.ceil(width / window_ratio))]
        # print(f">>>> {window_size = }")
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            if is_conv_block:
                nn = inverted_residual_block(nn, out_channel, **inverted_residual_block_kwargs, drop_rate=block_drop_rate, name=name)
                nn = activation_by_name(nn, activation=activation, name=name + "output_")
            else:
                nn = tiny_vit_block(nn, window_size, num_head, mlp_ratio, layer_scale, block_drop_rate, activation=activation, name=name)
            global_block_id += 1

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_output_")
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "tinyvit", pretrained, MultiHeadPositionalEmbedding)
    return model


def TinyViT_5M(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return TinyViT(**locals(), model_name=kwargs.pop("model_name", "tiny_vit_5m"), **kwargs)


def TinyViT_11M(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    out_channels = [64, 128, 256, 448]
    num_heads = [2, 4, 8, 14]
    return TinyViT(**locals(), model_name=kwargs.pop("model_name", "tiny_vit_11m"), **kwargs)


def TinyViT_21M(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    out_channels = [96, 192, 384, 576]
    num_heads = [3, 6, 12, 18]
    return TinyViT(**locals(), model_name=kwargs.pop("model_name", "tiny_vit_21m"), **kwargs)
