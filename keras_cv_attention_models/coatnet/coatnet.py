from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    se_module,
    output_block,
    MultiHeadRelativePositionalEmbedding,
    qkv_to_multi_head_channels_last_format,
    scaled_dot_product_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {"coatnet0": {"imagenet": {160: "bc4375d2f03b99ac4252770331f0d22f", 224: "5d1e563f959f7efb6d395bde7373ed26"}}}


def mhsa_with_multi_head_relative_position_embedding(
    inputs,
    num_heads=4,
    key_dim=0,
    global_query=None,
    out_shape=None,
    out_weight=True,
    qkv_bias=False,
    out_bias=False,
    attn_dropout=0,
    data_format=None,
    name=None,
):
    data_format = image_data_format() if data_format is None else data_format
    input_channel = inputs.shape[-1 if data_format == "channels_last" else 1]
    height, width = inputs.shape[1:-1] if data_format == "channels_last" else inputs.shape[2:]

    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    out_shape = input_channel if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    # vv_dim = out_shape // num_heads
    vv_dim = key_dim
    blocks = height * width

    # Permute for conv if given data_format not matching actual image_data_format
    if image_data_format() == "channels_last" and data_format == "channels_first":
        inputs = layers.Permute([2, 3, 1])(inputs)
    elif image_data_format() == "channels_first" and data_format == "channels_last":
        inputs = layers.Permute([3, 1, 2])(inputs)
    conv_channel_axis = -1 if image_data_format() == "channels_last" else 1

    if global_query is not None:
        # kv = layers.Dense(qk_out * 2, use_bias=qkv_bias, name=name and name + "kv")(inputs)  # For GCViT weights
        kv = conv2d_no_bias(inputs, qk_out * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "kv_")
        kv = functional.reshape(kv, [-1, blocks, kv.shape[-1]] if image_data_format() == "channels_last" else [-1, kv.shape[1], blocks])
        key, value = functional.split(kv, [qk_out, out_shape], axis=conv_channel_axis)
        query = global_query
        _, key, value = qkv_to_multi_head_channels_last_format(None, key, value, num_heads=num_heads, data_format=None)
    else:
        # qkv = conv2d_no_bias(inputs, qk_out * 2 + out_shape, use_bias=qkv_bias, kernel_size=1, name=name and name + "qkv_")
        # qkv = layers.Dense(qk_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)  # For GCViT weights
        # query = layers.Dense(qk_out, use_bias=qkv_bias, name=name and name + "query")(inputs)  # For MaxViT weights
        # key = layers.Dense(qk_out, use_bias=qkv_bias, name=name and name + "key")(inputs)  # For MaxViT weights
        # value = layers.Dense(qk_out, use_bias=qkv_bias, name=name and name + "value")(inputs)  # For MaxViT weights
        qkv = conv2d_no_bias(inputs, qk_out * 3, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
        query, key, value = functional.split(qkv, [qk_out, qk_out, qk_out], axis=conv_channel_axis)
        query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads=num_heads, data_format=None)
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }, {key_dim = }, {vv_dim = }")

    pos_emb = MultiHeadRelativePositionalEmbedding(with_cls_token=False, attn_height=height, name=name and name + "pos_emb")
    output_shape = (height, width, out_shape)
    out = scaled_dot_product_attention(query, key, value, output_shape, pos_emb, out_weight=out_weight, out_bias=out_bias, dropout=attn_dropout, name=name)
    return out if data_format == "channels_last" else layers.Permute([3, 1, 2], name=name and name + "output_perm")(out)


def res_MBConv(
    inputs,
    output_channel,
    conv_short_cut=True,
    strides=1,
    expansion=4,
    se_ratio=0,
    drop_rate=0,
    use_dw_strides=True,
    bn_act_first=False,
    activation="gelu",
    name="",
):
    """x ← Proj(Pool(x)) + Conv (DepthConv (Conv (Norm(x), stride = 2))))"""
    preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")

    if conv_short_cut:
        shortcut = layers.MaxPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
        # shortcut = batchnorm_with_activation(shortcut, activation=activation, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    # MBConv
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    conv_strides, dw_strides = (1, strides) if use_dw_strides else (strides, 1)  # May swap stirdes with DW
    nn = conv2d_no_bias(preact, input_channel * expansion, 1, strides=conv_strides, use_bias=bn_act_first, padding="same", name=name + "expand_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(nn, 3, strides=dw_strides, use_bias=bn_act_first, padding="same", name=name + "MB_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, zero_gamma=False, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, activation=activation, name=name + "se_")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="same", name=name + "MB_pw_")
    # nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "MB_pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return layers.Add(name=name + "output")([shortcut, nn])


def res_ffn(inputs, expansion=4, kernel_size=1, drop_rate=0, activation="gelu", name=""):
    """x ← x + Module (Norm(x)), similar with typical MLP block"""
    # preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")
    preact = layer_norm(inputs, name=name + "preact_")

    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = conv2d_no_bias(preact, input_channel * expansion, kernel_size, name=name + "1_")
    nn = activation_by_name(nn, activation=activation, name=name)
    nn = conv2d_no_bias(nn, input_channel, kernel_size, name=name + "2_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # return layers.Add(name=name + "output")([preact, nn])
    return layers.Add(name=name + "output")([inputs, nn])


def res_mhsa(inputs, output_channel, conv_short_cut=True, strides=1, head_dimension=32, drop_rate=0, activation="gelu", name=""):
    """x ← Proj(Pool(x)) + Attention (Pool(Norm(x)))"""
    # preact = batchnorm_with_activation(inputs, activation=None, zero_gamma=False, name=name + "preact_")
    preact = layer_norm(inputs, name=name + "preact_")

    if conv_short_cut:
        shortcut = layers.MaxPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
        # shortcut = batchnorm_with_activation(shortcut, activation=activation, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    nn = preact
    if strides != 1:  # Downsample
        # nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = layers.MaxPool2D(pool_size=2, strides=strides, padding="SAME", name=name + "pool")(nn)
    num_heads = nn.shape[-1 if image_data_format() == "channels_last" else 1] // head_dimension
    nn = mhsa_with_multi_head_relative_position_embedding(nn, num_heads=num_heads, key_dim=head_dimension, out_shape=output_channel, name=name + "mhsa_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{name = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return layers.Add(name=name + "output")([shortcut, nn])


def CoAtNet(
    num_blocks,
    out_channels,
    stem_width=64,
    block_types=["conv", "conv", "transform", "transform"],
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    use_dw_strides=True,
    bn_act_first=False,  # Experiment, use activation -> BatchNorm instead of BatchNorm -> activation, also set use_bias=True for pre Conv2D layer
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="coatnet",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, use_bias=bn_act_first, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=bn_act_first, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, use_bias=bn_act_first, padding="same", name="stem_2_")
    # nn = batchnorm_with_activation(nn, activation=activation, name="stem_2_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        stack_strides = strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        for block_id in range(num_block):
            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 else False
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block:
                nn = res_MBConv(
                    nn, out_channel, conv_short_cut, stride, expansion, block_se_ratio, block_drop_rate, use_dw_strides, bn_act_first, activation, name=name
                )
            else:
                nn = res_mhsa(nn, out_channel, conv_short_cut, stride, head_dimension, block_drop_rate, activation=activation, name=name)
                nn = res_ffn(nn, expansion=expansion, drop_rate=block_drop_rate, activation=activation, name=name + "ffn_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation, act_first=bn_act_first)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "coatnet", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def CoAtNetT(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", **kwargs):
    num_blocks = [3, 4, 6, 3]
    out_channels = [64, 128, 256, 512]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnett", **kwargs)


def CoAtNet0(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 5, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet0", **kwargs)


def CoAtNet1(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.3, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet1", **kwargs)


def CoAtNet2(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.5, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [128, 256, 512, 1024]
    stem_width = 128
    return CoAtNet(**locals(), model_name="coatnet2", **kwargs)


def CoAtNet3(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.7, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet3", **kwargs)


def CoAtNet4(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet4", **kwargs)


def CoAtNet5(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [256, 512, 1280, 2048]
    stem_width = 192
    head_dimension = 64
    return CoAtNet(**locals(), model_name="coatnet5", **kwargs)


def CoAtNet6(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [192, 384, 768, 1536, 2048]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet6", **kwargs)


def CoAtNet7(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0.2, classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [256, 512, 1024, 2048, 3072]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet7", **kwargs)
