import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    ChannelAffine,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mhsa_with_multi_head_relative_position_embedding,
    MultiHeadRelativePositionalEmbedding,
    se_module,
    output_block,
    window_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "maxvit_tiny": {"imagenet": {224: "e5cfd6a6bd4dea939860b6d8a29a911a", 384: "25b074ca6258d526c4d931f887599fe1", 512: "c745836c38f0ea98fef30bb3186aaf17"}},
    "maxvit_small": {"imagenet": {224: "6bbaff1c6316486c3ac29b607d9ebb13", 384: "85d2b77eab2f2645277a00b3f35d77d9", 512: "63b6c283ea6930fdcbe9b0e2f0bdbe6f"}},
    "maxvit_base": {
        "imagenet": {224: "00c833043b87ef2861ecf79820d827e0", 384: "1b6941018ed54944083add016780d9bd", 512: "530f48f41d49f3ad19cb2b7a5f790517"},
        "imagenet21k": {224: "9b89418259927f12b59df91bdaecae82"},
        "imagenet21k-ft1k": {384: "9b3f8c948e657450c856b0efb305dc7c", 512: "b0d554b7ea52a8bc87dc6790377f4672"},
    },
    "maxvit_large": {
        "imagenet": {224: "93d079fa8171986cc272f6fb4e9b0255", 384: "507afb8b65a99c4dd3c915e3525bdd03", 512: "8ac514a0e3534a24c05eb4087b75454b"},
        "imagenet21k": {224: "a3540387e39efaee5927e7fd9288cc0b"},
        "imagenet21k-ft1k": {384: "4063ab86cf515e7b0ab568cbc5a0726f", 512: "972a7c1cceacb8c1761338c5b6b72605"},
    },
    "maxvit_xlarge": {
        "imagenet21k": {224: "57574f4411cabff061b8946fb768a5df"},
        "imagenet21k-ft1k": {384: "6d78913bc04f716ab029bc6dbc0d9799", 512: "f79fc50827d10f43333321ab03fa563c"},
    },
}


def res_MBConv(inputs, output_channel, conv_short_cut=True, strides=1, expansion=4, se_ratio=0, use_torch_mode=False, drop_rate=0, activation="gelu", name=""):
    if use_torch_mode:
        use_torch_padding, epsilon, momentum = True, 1e-5, 0.9
    else:
        use_torch_padding, epsilon, momentum = False, 0.001, 0.99

    if strides > 1:
        shortcut = layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs)
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, use_bias=True, name=name + "shortcut_") if conv_short_cut else shortcut
    else:
        shortcut = inputs

    # MBConv
    preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, epsilon=epsilon, momentum=momentum, name=name + "preact_")
    nn = conv2d_no_bias(preact, output_channel * expansion, 1, strides=1, padding="same", name=name + "expand_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=epsilon, momentum=momentum, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(nn, 3, strides=strides, padding="SAME", use_torch_padding=use_torch_padding, name=name + "MB_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, epsilon=epsilon, momentum=momentum, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, activation="swish", name=name + "se/")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, use_bias=True, padding="same", name=name + "MB_pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{shortcut.shape = }, {nn.shape = }, {strides = }")
    return layers.Add(name=name + "output")([shortcut, nn])


def res_attn_ffn(inputs, output_channel, head_dimension=32, window_size=7, expansion=4, is_grid=False, drop_rate=0, layer_scale=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]  # Channels_last only
    attn = layer_norm(inputs, axis=-1, name=name + "attn_preact_")
    num_heads = attn.shape[-1] // head_dimension
    # print(f"{inputs.shape = }, {num_heads = }, {window_size = }, {is_grid = }")
    attention_block = lambda inputs, num_heads, name: mhsa_with_multi_head_relative_position_embedding(
        inputs, num_heads=num_heads, qkv_bias=True, out_bias=True, data_format="channels_last", name=name
    )
    attn = window_attention(attn, window_size=window_size, num_heads=num_heads, is_grid=is_grid, attention_block=attention_block, name=name + "window_mhsa/")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    # print(f"{name = }, {inputs.shape = }, {inputs.shape = }, {attn.shape = }")
    attn = layers.Add(name=name + "attn_output")([inputs, attn])

    ffn = layer_norm(attn, axis=-1, name=name + "ffn_preact_")
    ffn = layers.Dense(input_channel * expansion, name=name + "ffn/1_dense")(ffn)
    ffn = activation_by_name(ffn, activation=activation, name=name)
    ffn = layers.Dense(input_channel, name=name + "ffn/2_dense")(ffn)
    ffn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "2_gamma")(ffn) if layer_scale >= 0 else ffn
    ffn = drop_block(ffn, drop_rate=drop_rate, name=name + "ffn_")
    return layers.Add(name=name + "ffn_output")([attn, ffn])


def MaxViT(
    num_blocks,
    out_channels,
    stem_width=64,
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    window_ratio=32,
    output_filter=-1,  # -1 for out_channels[-1], 0 to disable
    use_torch_mode=False,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu/app",  # means tf.nn.gelu(approximate=True)
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="maxvit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    height, width = inputs.shape[1:-1] if image_data_format() == "channels_last" else inputs.shape[2:]
    if use_torch_mode:
        use_torch_padding, epsilon, momentum = True, 1e-5, 0.9
    else:
        use_torch_padding, epsilon, momentum = False, 0.001, 0.99

    """ Stem """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, use_bias=True, padding="same", use_torch_padding=use_torch_padding, name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=epsilon, momentum=momentum, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, use_bias=True, padding="same", use_torch_padding=use_torch_padding, name="stem_2_")
    window_size = [int(math.ceil(height / window_ratio)), int(math.ceil(width / window_ratio))]
    # print(f"{window_size = }")

    attn_ffn_common_kwargs = {
        "head_dimension": head_dimension,
        "window_size": window_size,
        "expansion": expansion,
        "layer_scale": layer_scale,
        "activation": activation,
    }

    """ Backbone [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        stack_strides = strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        for block_id in range(num_block):
            name = "stack_{}_block_{}/".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 and nn.shape[channel_axis] != out_channel else False
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = res_MBConv(
                nn, out_channel, conv_short_cut, stride, expansion, block_se_ratio, use_torch_mode, block_drop_rate, activation, name=name + "mbconv/"
            )
            nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
            nn = res_attn_ffn(nn, out_channel, is_grid=False, drop_rate=block_drop_rate, name=name + "block_", **attn_ffn_common_kwargs)
            nn = res_attn_ffn(nn, out_channel, is_grid=True, drop_rate=block_drop_rate, name=name + "grid_", **attn_ffn_common_kwargs)
            nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layer_norm(nn, name="post_")
        output_filter = out_channels[-1] if output_filter == -1 else output_filter
        if output_filter > 0:
            nn = layers.Dense(output_filter, name="features")(nn)
            nn = activation_by_name(nn, "tanh", name="features_")
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    rescale_mode = "tf" if pretrained is not None and pretrained.startswith("imagenet21k") else "torch"  # For testing only
    add_pre_post_process(model, rescale_mode=rescale_mode)
    reload_model_weights(model, PRETRAINED_DICT, "maxvit", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def MaxViT_Tiny(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 5, 2]
    out_channels = [64, 128, 256, 512]
    stem_width = 64
    return MaxViT(**locals(), model_name="maxvit_tiny", **kwargs)


def MaxViT_Small(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 5, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return MaxViT(**locals(), model_name="maxvit_small", **kwargs)


def MaxViT_Base(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return MaxViT(**locals(), model_name="maxvit_base", **kwargs)


def MaxViT_Large(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [128, 256, 512, 1024]
    stem_width = 128
    return MaxViT(**locals(), model_name="maxvit_large", **kwargs)


def MaxViT_XLarge(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return MaxViT(**locals(), model_name="maxvit_xlarge", **kwargs)
