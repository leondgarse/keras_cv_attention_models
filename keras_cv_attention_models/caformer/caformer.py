from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    conv2d_no_bias,
    layer_norm,
    mlp_block,
    mlp_block_with_depthwise_conv,
    multi_head_self_attention,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "caformer_s18": {
        "imagenet": {224: "b8a824d4197161286de2aef08c1be379", 384: "fcd11441f1811124f77a4cad1b328639"},
        "imagenet21k-ft1k": {224: "4b87b7d0393dc607089eff549dfe7319", 384: "05f20bcd5403a6076b1fa3276766b987"},
    },
    "caformer_s36": {
        "imagenet": {224: "df683079147328b5cafeb6e98e99a885", 384: "532becc7aaa087430baa1691ed2e64eb"},
        "imagenet21k-ft1k": {224: "89b40eef0fc073932460e2ba8094acb7", 384: "f4d163c51c8f06f452cd0f312ee51c3d"},
    },
    "caformer_m36": {
        "imagenet": {224: "04a88000f78cc8c1e9b261dc71fa3e6f", 384: "3c3b13ffb3cfbd304897abaef5fb0abf"},
        "imagenet21k-ft1k": {224: "af5a2978cc60609ed70941f506d4a337", 384: "3055798b1c0f291547911460390c1964"},
    },
    "caformer_b36": {
        "imagenet": {224: "8461e68f9dde4fe6ef201874e4d6cb66", 384: "087bddb30c6f04a44e4b3a08864f2327"},
        "imagenet21k-ft1k": {224: "47f77c5087b4acd427ec960074136644", 384: "a6c78cb35254870a475487b4a582bcd1"},
    },
    "convformer_s18": {
        "imagenet": {224: "ef64c846c5890258227281fc6d2dedf7", 384: "aa8d9c6e6f47e3934ed329ded7f85674"},
        "imagenet21k-ft1k": {224: "da24d177fc47d878ae3621be82130ede", 384: "1e8ba700dc67840ecde6c76f0688d2f4"},
    },
    "convformer_s36": {
        "imagenet": {224: "2eee80c9126a08cb09c386de7d9759b6", 384: "a8f175274494598cb322ed0c5f17e4b9"},
        "imagenet21k-ft1k": {224: "e75e739dc0f09dbd88b326520e4b7fb0", 384: "2f9bfe0664ff3128820d2873915709a1"},
    },
    "convformer_m36": {
        "imagenet": {224: "2e7a8c5f9827b0ca8fe51f9a0aa92223", 384: "3a53ed969febefb1d080ce0b78dcbb33"},
        "imagenet21k-ft1k": {224: "652c8ec792c50a0f87471c5d53e92940", 384: "f37c31964b650a7700e7820ea4ccf69e"},
    },
    "convformer_b36": {
        "imagenet": {224: "c96d0f4720c36ae19ab8eee02c6e6034", 384: "109236ff75aabe1885ef625c3bfe756c"},
        "imagenet21k-ft1k": {224: "64774d61660d2e95df8d274785f7708a", 384: "b7e350d127a37ddfa7cd611ce21b4e4b"},
    },
}


def meta_former_block(inputs, use_attn=False, head_dim=32, mlp_ratio=4, layer_scale=0, residual_scale=0, drop_rate=0, activation="star_relu", name=""):
    # channnel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[-1]

    """ attention """
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, center=False, axis=-1, name=name + "attn_")
    # nn = conv_pool_attention_mixer(nn, num_heads, num_attn_low_heads=num_attn_low_heads, pool_size=pool_size, activation=activation, name=name + "attn_")
    if use_attn:
        nn = multi_head_self_attention(nn, num_heads=input_channel // head_dim, name=name + "mhsa_")
    else:
        nn = mlp_block_with_depthwise_conv(nn, input_channel * 2, kernel_size=7, use_bias=False, activation=(activation, None), name=name + "mlp_sep_")
    attn_out = add_with_layer_scale_and_drop_block(
        inputs, nn, layer_scale=layer_scale, residual_scale=residual_scale, drop_rate=drop_rate, axis=-1, name=name + "attn_"
    )

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, center=False, axis=-1, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, use_bias=False, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(
        attn_out, nn, layer_scale=layer_scale, residual_scale=residual_scale, drop_rate=drop_rate, axis=-1, name=name + "mlp_"
    )
    return nn


def CAFormer(
    num_blocks=[3, 3, 9, 3],
    out_channels=[64, 128, 320, 512],
    block_types=["conv", "conv", "transform", "transform"],
    head_dim=32,
    mlp_ratios=4,
    head_filter=2048,
    head_filter_activation="squared_relu",
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="star_relu",
    drop_connect_rate=0,
    dropout=0,
    layer_scales=0,
    residual_scales=[0, 0, 1, 1],
    classifier_activation="softmax",
    pretrained=None,
    model_name="caformer",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = layers.ZeroPadding2D(padding=2, name="stem_")(inputs)  # padding=2
    nn = conv2d_no_bias(nn, out_channels[0], kernel_size=7, strides=4, padding="valid", use_bias=True, name="stem_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name="stem_pre_permute")(nn)
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, center=False, axis=-1, name="stem_")

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, center=False, axis=-1, name=stack_name + "downsample_")
            nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=stack_name + "permute_pre")(nn)
            nn = conv2d_no_bias(nn, out_channel, 3, strides=2, padding="same", use_bias=True, name=stack_name + "downsample_")
            nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=stack_name + "permute_post")(nn)

        use_attn = True if block_type[0].lower() == "t" else False
        mlp_ratio = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        layer_scale = layer_scales[stack_id] if isinstance(layer_scales, (list, tuple)) else layer_scales
        residual_scale = residual_scales[stack_id] if isinstance(residual_scales, (list, tuple)) else residual_scales
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = meta_former_block(nn, use_attn, head_dim, mlp_ratio, layer_scale, residual_scale, block_drop_rate, activation=activation, name=name)
            global_block_id += 1
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name="pre_output_permute")(nn)

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_output_")
        if head_filter > 0:
            nn = layers.Dense(head_filter, use_bias=True, name="feature_dense")(nn)
            head_filter_activation = head_filter_activation if head_filter_activation is not None else activation
            nn = activation_by_name(nn, activation=head_filter_activation, name="feature_")
            nn = layer_norm(nn, name="feature_")  # epsilon=1e-5
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "caformer", pretrained)
    return model


def CAFormerS18(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    kwargs.pop("kwargs", None)
    return CAFormer(**locals(), model_name=kwargs.pop("model_name", "caformer_s18"), **kwargs)


def CAFormerS36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 12, 18, 3]
    kwargs.pop("kwargs", None)
    return CAFormer(**locals(), model_name=kwargs.pop("model_name", "caformer_s36"), **kwargs)


def CAFormerM36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 12, 18, 3]
    out_channels = [96, 192, 384, 576]
    head_filter = out_channels[-1] * 4
    kwargs.pop("kwargs", None)
    return CAFormer(**locals(), model_name=kwargs.pop("model_name", "caformer_m36"), **kwargs)


def CAFormerB36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 12, 18, 3]
    out_channels = [128, 256, 512, 768]
    head_filter = out_channels[-1] * 4
    kwargs.pop("kwargs", None)
    return CAFormer(**locals(), model_name=kwargs.pop("model_name", "caformer_b36"), **kwargs)


""" ConvFormer """


def ConvFormerS18(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CAFormerS18(**locals(), block_types=["conv", "conv", "conv", "conv"], model_name="convformer_s18", **kwargs)


def ConvFormerS36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CAFormerS36(**locals(), block_types=["conv", "conv", "conv", "conv"], model_name="convformer_s36", **kwargs)


def ConvFormerM36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CAFormerM36(**locals(), block_types=["conv", "conv", "conv", "conv"], model_name="convformer_m36", **kwargs)


def ConvFormerB36(input_shape=(224, 224, 3), num_classes=1000, activation="star_relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CAFormerB36(**locals(), block_types=["conv", "conv", "conv", "conv"], model_name="convformer_b36", **kwargs)
