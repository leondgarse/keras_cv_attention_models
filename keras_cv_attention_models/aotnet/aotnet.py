import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import os
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    anti_alias_downsample,
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    se_module,
    eca_module,
    # output_block,
    add_pre_post_process,
)
from keras_cv_attention_models import attention_layers

# from keras_cv_attention_models.download_and_load import reload_model_weights

DEFAULT_PARAMS = {
    "bot": {"num_heads": 4, "relative": True, "out_bias": False},
    "halo": {"num_heads": 8, "block_size": 4, "halo_size": 1},
    "sa": {"kernel_size": 3, "groups": 2, "downsample_first": False},
    "cot": {"kernel_size": 3, "downsample_first": True},
    "outlook": {"num_heads": 8, "kernel_size": 3},
}


def attn_block(
    inputs,
    filters,
    strides=1,
    attn_type=None,
    attn_params={},
    se_ratio=0,
    use_eca=False,
    groups=1,
    group_size=0,
    bn_after_attn=True,
    use_evo_norm=False,
    evo_norm_group_size=-1,
    epsilon=1e-5,
    activation="relu",
    name="",
):
    nn = inputs
    if attn_params is not None and len(attn_params) != 0:
        default_attn_params = DEFAULT_PARAMS.get(attn_type, {}).copy()
        default_attn_params.update(attn_params)
        attn_params = default_attn_params
    else:
        attn_params = DEFAULT_PARAMS.get(attn_type, {})

    attn_act = attn_params.pop("activation", activation)
    se_divisor = attn_params.pop("se_divisor", 8)
    if attn_type == "bot":  # mhsa_with_relative_position_embedding from botnet
        nn = attention_layers.mhsa_with_relative_position_embedding(nn, **attn_params, name=name + "mhsa_")
    elif attn_type == "halo":  # halo_attention from halonet
        halo_expansion = attn_params.pop("halo_expansion", 1)
        out_shape = int(filters * halo_expansion)
        nn = attention_layers.halo_attention(nn, **attn_params, strides=strides, out_shape=out_shape, name=name + "halo")
    elif attn_type == "sa":  # split_attention_conv2d from resnest
        nn = attention_layers.split_attention_conv2d(nn, **attn_params, filters=filters, strides=strides, activation=attn_act, name=name + "sa_")
    elif attn_type == "cot":  # cot_attention from cotnet
        nn = attention_layers.cot_attention(nn, **attn_params, strides=strides, activation=attn_act, name=name + "cot_")
    elif attn_type == "outlook":  # outlook_attention from volo
        nn = attention_layers.outlook_attention(nn, filters, **attn_params, name=name + "outlook_")
    # elif attn_type == "groups_conv":  # ResNeXt like
    #     nn = conv2d_no_bias(nn, filters, **attn_params, strides=strides, padding="SAME", name=name + "GC_")
    else:  # ResNet and `groups > 1` for ResNeXt like
        groups = groups if group_size == 0 else filters // group_size
        conv_name = (name + "GC_") if groups > 1 else name
        nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", groups=groups, name=conv_name)

    if attn_type in ["bot", "outlook"] and strides != 1:  # Downsample
        # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.AveragePooling2D(pool_size=2, strides=strides, name=name + "pool")(nn)

    if bn_after_attn:
        bn_params = {"use_evo_norm": use_evo_norm, "evo_norm_group_size": evo_norm_group_size, "epsilon": epsilon}
        nn = batchnorm_with_activation(nn, activation, zero_gamma=False, **bn_params, name=name)

    if attn_type is None and se_ratio:
        nn = se_module(nn, se_ratio=se_ratio, divisor=se_divisor, activation=attn_act, name=name + "se_")

    if attn_type is None and use_eca:
        nn = eca_module(nn, name=name + "eca_")
    return nn


def conv_shortcut_branch(inputs, filters, preact=False, strides=1, shortcut_type="conv", bn_act_params={}, name=""):
    if shortcut_type is None:
        return None

    if strides > 1 and shortcut_type == "avg":
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_down")(inputs)
        strides = 1
    elif strides > 1 and shortcut_type == "anti_alias":
        shortcut = anti_alias_downsample(inputs, kernel_size=3, strides=2, name=name + "shortcut_down")
        strides = 1
    else:
        shortcut = inputs
    shortcut = conv2d_no_bias(shortcut, filters, 1, strides=strides, name=name + "shortcut_")
    if not preact:  # ResNet
        shortcut = batchnorm_with_activation(shortcut, zero_gamma=False, **bn_act_params, name=name + "shortcut_")
    return shortcut


def deep_branch(
    inputs, filters, strides=1, hidden_channel_ratio=0.25, use_3x3_kernel=False, bn_after_attn=True, bn_act_params={}, attn_block_params={}, name=""
):
    hidden_filter = int(filters * hidden_channel_ratio)
    if use_3x3_kernel:
        nn = conv2d_no_bias(inputs, hidden_filter, 3, strides=1, padding="SAME", name=name + "deep_1_")  # Using strides=1 for not changing input shape
        # nn = conv2d_no_bias(inputs, hidden_filter, 3, strides=strides, padding="SAME", name=name + "1_")
        # strides = 1
    else:
        nn = conv2d_no_bias(inputs, hidden_filter, 1, strides=1, padding="VALID", name=name + "deep_1_")
    nn = batchnorm_with_activation(nn, zero_gamma=False, **bn_act_params, name=name + "deep_1_")
    # bn_after_attn = False if use_3x3_kernel else bn_after_attn
    nn = attn_block(nn, hidden_filter, strides, **attn_block_params, **bn_act_params, bn_after_attn=bn_after_attn, name=name + "deep_2_")

    if not use_3x3_kernel:
        nn = conv2d_no_bias(nn, filters, 1, strides=1, padding="VALID", name=name + "deep_3_")
    return nn


def aot_block(
    inputs,
    filters,
    strides=1,
    conv_shortcut=False,
    hidden_channel_ratio=0.25,
    drop_rate=0,
    preact=False,
    use_3x3_kernel=False,
    bn_after_attn=True,
    shortcut_type="conv",
    use_block_output_activation=True,
    use_evo_norm=False,
    epsilon=1e-5,
    evo_norm_group_size=-1,
    activation="relu",
    attn_block_params={},
    name="",
):
    if attn_block_params.get("attn_type", None) == "halo":  # HaloAttention
        halo_block_size = attn_block_params.get("attn_params", {}).get("block_size", DEFAULT_PARAMS["halo"]["block_size"])
        if inputs.shape[1] % halo_block_size != 0 or inputs.shape[2] % halo_block_size != 0:
            gap_h = halo_block_size - inputs.shape[1] % halo_block_size
            gap_w = halo_block_size - inputs.shape[2] % halo_block_size
            pad_head_h, pad_tail_h = gap_h // 2, gap_h - gap_h // 2
            pad_head_w, pad_tail_w = gap_w // 2, gap_w - gap_w // 2
            # print(f">>>> Halo pad: {inputs.shape = }, {pad_head_h = }, {pad_tail_h = }, {pad_head_w = }, {pad_tail_w = }")
            inputs = keras.layers.ZeroPadding2D(padding=((pad_head_h, pad_tail_h), (pad_head_w, pad_tail_w)), name=name + "gap_pad")(inputs)

    bn_params = {"use_evo_norm": use_evo_norm, "evo_norm_group_size": evo_norm_group_size, "epsilon": epsilon}
    bn_act_params = {"activation": activation, "use_evo_norm": use_evo_norm, "evo_norm_group_size": evo_norm_group_size, "epsilon": epsilon}
    if preact:  # ResNetV2
        pre_inputs = batchnorm_with_activation(inputs, zero_gamma=False, **bn_act_params, name=name + "preact_")
    else:
        pre_inputs = inputs

    if conv_shortcut:  # Set a new shortcut using conv
        # short_act = activation if attn_block_params["attn_type"] == "bot" else None
        shortcut = conv_shortcut_branch(pre_inputs, filters, preact, strides, shortcut_type, bn_params, name=name)  # activation=None
    else:
        shortcut = keras.layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs

    deep = deep_branch(pre_inputs, filters, strides, hidden_channel_ratio, use_3x3_kernel, bn_after_attn, bn_act_params, attn_block_params, name=name)

    # print(f">>>> {inputs.shape = }, {shortcut if shortcut is None else shortcut.shape = }, {deep.shape = }, {filters = }, {strides = }")
    if preact:  # ResNetV2
        deep = drop_block(deep, drop_rate)
        return keras.layers.Add(name=name + "output")([shortcut, deep]) if shortcut is not None else deep  # if no shortcut
    else:
        if not (use_3x3_kernel and bn_after_attn):
            deep = batchnorm_with_activation(deep, activation=None, zero_gamma=True, **bn_params, name=name + "3_")
        deep = drop_block(deep, drop_rate)
        out = keras.layers.Add(name=name + "add")([shortcut, deep]) if shortcut is not None else deep  # if no shortcut
        if use_block_output_activation:
            out = activation_by_name(out, activation, name=name + "out_")
        return keras.layers.Activation("linear", name=name + "output")(out)  # Identity, Just need a name here


def aot_stack(
    inputs,
    blocks,
    filters,
    strides=2,
    strides_first=True,
    hidden_channel_ratio=0.25,
    stack_drop=0,
    block_params={},
    attn_types=None,
    attn_params={},
    se_ratio=0,
    use_eca=False,
    groups=1,
    group_size=0,
    name="",
):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    strides_block_id = 0 if strides_first else blocks - 1
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters) else False
        cur_strides = strides if id == strides_block_id else 1
        block_name = name + "block{}_".format(id + 1)
        block_drop_rate = stack_drop[id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        cur_ratio = hidden_channel_ratio[id] if isinstance(hidden_channel_ratio, (list, tuple)) else hidden_channel_ratio
        attn_block_params = {  # Just save the line width..
            "attn_type": attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types,
            "attn_params": attn_params[id] if isinstance(attn_params, (list, tuple)) else attn_params,
            "se_ratio": se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio,
            "use_eca": use_eca[id] if isinstance(use_eca, (list, tuple)) else use_eca,
            "groups": groups[id] if isinstance(groups, (list, tuple)) else groups,
            "group_size": group_size[id] if isinstance(group_size, (list, tuple)) else group_size,
        }
        nn = aot_block(
            nn, filters, cur_strides, conv_shortcut, cur_ratio, block_drop_rate, **block_params, attn_block_params=attn_block_params, name=block_name
        )
    return nn


def deep_stem(inputs, stem_width, use_half_channel=True, activation="relu", last_strides=1, bn_params={}, name=None):
    hidden_channel = stem_width // 2 if use_half_channel else stem_width
    nn = conv2d_no_bias(inputs, hidden_channel, 3, strides=2, padding="same", name=name and name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "1_")
    nn = conv2d_no_bias(nn, hidden_channel, 3, strides=1, padding="same", name=name and name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "3_")
    return nn


def quad_stem(inputs, stem_width, activation="relu", stem_act=False, last_strides=2, bn_params={}, name=None):
    nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name=name and name + "1_")
    if stem_act:
        nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=1, padding="same", name=name and name + "2_")
    if stem_act:
        nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name and name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "3_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "4_")
    return nn


def tiered_stem(inputs, stem_width, activation="relu", last_strides=1, bn_params={}, name=None):
    nn = conv2d_no_bias(inputs, 3 * stem_width // 8, 3, strides=2, padding="same", name=name and name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name and name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "3_")
    return nn


def aot_stem(inputs, stem_width, type=None, activation="relu", quad_stem_act=False, last_strides=1, bn_params={}, name=None):
    """ stem_type in value `[None, "deep", "deep2", "quad", "tiered"]`. """
    if type == "deep":
        nn = deep_stem(inputs, stem_width, activation=activation, last_strides=last_strides, bn_params=bn_params, name=name)
    elif type == "deep2":  # RegNetZD8_EVO
        nn = deep_stem(inputs, stem_width, use_half_channel=False, activation=activation, last_strides=last_strides, bn_params=bn_params, name=name)
    elif type == "quad":
        nn = quad_stem(inputs, stem_width, activation=activation, stem_act=quad_stem_act, last_strides=last_strides, bn_params=bn_params, name=name)
    elif type == "tiered":
        nn = tiered_stem(inputs, stem_width, activation=activation, last_strides=last_strides, bn_params=bn_params, name=name)
    elif type == "kernel_3x3":
        nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name=name)
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name)
    return nn


def AotNet(
    num_blocks,  # Stack parameters
    preact=False,  # False for resnet, True for resnetv2
    strides=[1, 2, 2, 2],
    strides_first=True,  # True for resnet, False for resnetv2
    out_channels=[256, 512, 1024, 2048],
    hidden_channel_ratio=0.25,
    use_3x3_kernel=False,
    use_block_output_activation=True,
    use_evo_norm=False,
    bn_epsilon=1e-5,
    evo_norm_group_size=-1,
    stem_width=64,  # Stem params
    stem_type=None,  # ["deep", "quad", "tiered", "kernel_3x3", None]
    quad_stem_act=False,
    stem_last_strides=1,
    stem_downsample=True,
    attn_types=None,  # Attention block params
    attn_params={},
    se_ratio=0,  # (0, 1)
    use_eca=False,
    groups=1,
    group_size=0,
    bn_after_attn=True,
    shortcut_type="conv",  # shortcut_branch params, ["conv", "avg", "anti_alias", None]
    input_shape=(224, 224, 3),  # Model common params
    num_classes=1000,
    activation="relu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    output_num_features=0,
    dropout=0,
    model_name="aotnet",
    pretrained=None,
    kwargs=None,
):
    """ Stem """
    inputs = keras.layers.Input(shape=input_shape)
    bn_params = {"use_evo_norm": use_evo_norm, "evo_norm_group_size": evo_norm_group_size, "epsilon": bn_epsilon}
    nn = aot_stem(inputs, stem_width, stem_type, activation, quad_stem_act, last_strides=stem_last_strides, bn_params=bn_params, name="stem_")

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name="stem_")
    if stem_downsample:
        nn = keras.layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    """ Stacks """
    block_params = {  # params same for all blocks
        "preact": preact,
        "use_3x3_kernel": use_3x3_kernel,
        "use_block_output_activation": use_block_output_activation,
        "bn_after_attn": bn_after_attn,
        "shortcut_type": shortcut_type,
        "use_evo_norm": use_evo_norm,
        "evo_norm_group_size": evo_norm_group_size,
        "epsilon": bn_epsilon,
        "activation": activation,
    }

    drop_connect_rates = drop_connect_rates_split(num_blocks, start=0.0, end=drop_connect_rate)
    for id, (num_block, out_channel, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, strides, drop_connect_rates)):
        name = "stack{}_".format(id + 1)
        cur_attn_params = {
            "attn_types": attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types,
            "attn_params": attn_params[id] if isinstance(attn_params, (list, tuple)) else attn_params,
            "se_ratio": se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio,
            "use_eca": use_eca[id] if isinstance(use_eca, (list, tuple)) else use_eca,
            "groups": groups[id] if isinstance(groups, (list, tuple)) else groups,
            "group_size": group_size[id] if isinstance(group_size, (list, tuple)) else group_size,
        }
        cur_ratio = hidden_channel_ratio[id] if isinstance(hidden_channel_ratio, (list, tuple)) else hidden_channel_ratio
        nn = aot_stack(nn, num_block, out_channel, stride, strides_first, cur_ratio, drop_connect, block_params, **cur_attn_params, name=name)

    if preact:  # resnetv2 like
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, **bn_params, name="post_")

    """ Output """
    # nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    if output_num_features > 0:
        nn = conv2d_no_bias(nn, output_num_features, 1, strides=1, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, **bn_params, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    # reload_model_weights(model, pretrained_dict={}, sub_release="aotnet", input_shape=input_shape, pretrained=pretrained)
    return model


def AotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name=kwargs.pop("model_name", "aotnet50"), **kwargs)


def AotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name=kwargs.pop("model_name", "aotnet101"), **kwargs)


def AotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name=kwargs.pop("model_name", "aotnet152"), **kwargs)


def AotNet200(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name=kwargs.pop("model_name", "aotnet200"), **kwargs)


def AotNetV2(num_blocks, preact=True, strides=1, strides_first=False, **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(num_blocks, preact=preact, strides=strides, strides_first=strides_first, **kwargs)


def AotNet50V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name=kwargs.pop("model_name", "aotnet50v2"), **kwargs)


def AotNet101V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name=kwargs.pop("model_name", "aotnet101v2"), **kwargs)


def AotNet152V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name=kwargs.pop("model_name", "aotnet152v2"), **kwargs)


def AotNet200V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=2, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name=kwargs.pop("model_name", "aotnet200v2"), **kwargs)
