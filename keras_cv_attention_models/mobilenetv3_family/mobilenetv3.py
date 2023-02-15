from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    inverted_residual_block,
    make_divisible,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "fbnetv3_b": {"imagenet": "498af18b2241fa07e029fca393885d93"},
    "fbnetv3_d": {"imagenet": "086ddd9ccf516f3699e80f06f231d41c"},
    "fbnetv3_g": {"imagenet": "5ed06c04bfeeff798e2d951ec863214c"},
    "lcnet_050": {"imagenet": "f740b5d73a42f65c5f7baf436d75d75f", "ssld": "2bfa67fa829e799710d76880df485ceb"},
    "lcnet_075": {"imagenet": "c2dd36d7362a782e70d7b5c7b4422027"},
    "lcnet_100": {"imagenet": "cb349c3f11678abae2799b88163fc614", "ssld": "ac978d02c94be99e4c6c10c24abfc4a8"},
    "lcnet_150": {"imagenet": "c1d114d56e88ff67e5ae4222db64c4fb"},
    "lcnet_200": {"imagenet": "46a7426993ec2910b93eab479528af68"},
    "lcnet_250": {"imagenet": "6dbe4e7bd1dd7a3ec227e2140f2ea162", "ssld": "ca752dc4ee870201a9a12d2234f284b2"},
    "mobilenetv3_large_075": {"imagenet": "56dc3594efb27c1df2c679d5486b9379"},
    "mobilenetv3_large_100": {
        "imagenet": "e5dbb4947d1fac0e5b0f90c7b6c3b6e9",
        "miil_21k": "86d0e21b372ff02498062f55e253b61c",
        "miil": "b2cb65b167a16a54795d87f1aaf0bf1e",
    },
    "mobilenetv3_small_050": {"imagenet": "17063e82099f420d552cc5f06efd9b46"},
    "mobilenetv3_small_075": {"imagenet": "1df2126a9ed19704996d969f57afa7bb"},
    "mobilenetv3_small_100": {"imagenet": "aa84f2bb4d7faf9fe2417267d2fc35b1"},
    "tinynet_a": {"imagenet": "ccffe4208feb4e9573834ab9cd4074e8"},
    "tinynet_b": {"imagenet": "4888c408c2ac69bf7824da383c58d52d"},
    "tinynet_c": {"imagenet": "29ce15979b8800176621780b0ea91eaa"},
    "tinynet_d": {"imagenet": "e08615d88e71f1548d040e697926514e"},
    "tinynet_e": {"imagenet": "514fbbcd582db6d4297a140efb84af9a"},
}


def avg_pool_conv_output(inputs, output_num_features=1280, use_output_feature_bias=True, activation="hard_swish"):
    # nn = layers.AveragePooling2D(pool_size=inputs.shape[1:3], name="avg_pool")(inputs)
    # nn = layers.GlobalAveragePooling2D(name="avg_pool")(inputs)[:, None, None, :]
    h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]
    nn = functional.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)  # using AvgPool2D cannot set dynamic input_shape
    if output_num_features > 0:
        nn = conv2d_no_bias(nn, make_divisible(output_num_features, 8), use_bias=use_output_feature_bias, name="features_")
        nn = activation_by_name(nn, activation, name="features_")
    nn = layers.Flatten()(nn)
    return nn


def conv_avg_pool_output(inputs, output_num_features=1280, use_output_feature_bias=True, activation="hard_swish"):
    nn = inputs
    if output_num_features > 0:
        nn = conv2d_no_bias(nn, make_divisible(output_num_features, 8), use_bias=use_output_feature_bias, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, name="features_")
    nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
    return nn


def MobileNetV3(
    num_blocks=[1, 2, 3, 4, 2, 3],  # [Stack parameters]
    out_channels=[16, 24, 40, 80, 112, 160],
    expands=[1, [4, 3], 3, [6, 2.5, 2.3, 2.3], 6, 6],
    kernel_sizes=[3, 3, 5, 3, 3, 5],
    strides=[1, 2, 2, 2, 1, 2],
    activations=["relu", "relu", "relu", "hard_swish", "hard_swish", "hard_swish"],
    disable_shortcut=False,  # True for LCNet
    use_blocks_output_activation=False,  # True for LCNet
    width_ratio=1.0,
    stem_width=16,  # [Stem parameters]
    fix_stem=False,
    stem_feature_activation="hard_swish",  # "swish" for TinyNet
    se_ratios=[0, 0, 0.25, 0, 0.25, 0.25],  # [SE module parameters]
    se_activation=("relu", "hard_sigmoid_torch"),  # ("hard_swish", "hard_sigmoid_torch") for FBNetV3, None for TinyNet
    se_limit_round_down=0.9,  # 0.95 for FBNetV3
    se_divisor=8,  # 1 for TinyNet
    use_expanded_se_ratio=True,  # False for FBNetV3, TinyNet
    output_num_features=1280,  # [Output parameters]
    use_additional_output_conv=True,  # False for LCNet, TinyNet
    use_output_feature_bias=True,  # False for FBNetV3, TinyNet
    use_avg_pool_conv_output=True,  # False for TinyNet
    input_shape=(224, 224, 3),  # [Model common parameters]
    num_classes=1000,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="mobilenetv3",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    stem_width = stem_width if fix_stem else make_divisible(stem_width * width_ratio, 8)
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=stem_feature_activation, name="stem_")

    block_kwargs = {  # Common parameters for all blocks
        "is_torch_mode": True,
        "se_activation": se_activation,
        "se_divisor": se_divisor,
        "se_limit_round_down": se_limit_round_down,
    }

    """ stage [1, 2, 3, 4] """
    pre_out = stem_width
    global_block_id = 0
    total_blocks = sum(num_blocks)
    for id, (num_block, out_channel, kernel_size, stride) in enumerate(zip(num_blocks, out_channels, kernel_sizes, strides)):
        stage_name = "stack{}_".format(id + 1)
        out_channel = make_divisible(out_channel * width_ratio, 8)
        activation = activations[id] if isinstance(activations, (list, tuple)) else activations
        expand = expands[id] if isinstance(expands, (list, tuple)) else expands
        se_ratio = se_ratios[id] if isinstance(se_ratios, (list, tuple)) else se_ratios
        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            stride = stride if block_id == 0 else 1
            shortcut = True if out_channel == pre_out and stride == 1 and not disable_shortcut else False
            cur_expand = expand[min(block_id, len(expand) - 1)] if isinstance(expand, (list, tuple)) else expand
            cur_se_ratio = se_ratio * cur_expand if use_expanded_se_ratio else se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks

            nn = inverted_residual_block(
                nn, out_channel, stride, cur_expand, shortcut, kernel_size, block_drop_rate, cur_se_ratio, **block_kwargs, activation=activation, name=name
            )
            if use_blocks_output_activation:
                nn = activation_by_name(nn, activation, name=name + "output_")
            pre_out = out_channel
            global_block_id += 1

    if use_additional_output_conv:
        nn = conv2d_no_bias(nn, make_divisible(out_channels[-1] * cur_expand * width_ratio, 8), kernel_size=1, name="pre_output_")
        nn = batchnorm_with_activation(nn, activation=stem_feature_activation, name="pre_output_")

    if num_classes > 0:
        if use_avg_pool_conv_output:
            nn = avg_pool_conv_output(nn, output_num_features, use_output_feature_bias, stem_feature_activation)
        else:
            nn = conv_avg_pool_output(nn, output_num_features, use_output_feature_bias, stem_feature_activation)

        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "mobilenetv3_family", pretrained)
    return model


def MobileNetV3Large(model_name="mobilenetv3_large", **kwargs):
    kwargs.pop("kwargs", None)
    return MobileNetV3(**locals(), **kwargs)


def MobileNetV3Small(
    num_blocks=[1, 2, 3, 2, 3],
    out_channels=[16, 24, 40, 48, 96],
    expands=[1, [4.5, 3.67], [4, 6, 6], 3, 6],
    kernel_sizes=[3, 3, 5, 5, 5],
    strides=[2, 2, 2, 1, 2],
    activations=["relu", "relu", "hard_swish", "hard_swish", "hard_swish"],
    se_ratios=[0.25, 0, 0.25, 0.25, 0.25],
    output_num_features=1024,
    model_name="mobilenetv3_small",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return MobileNetV3(**locals(), **kwargs)


def MobileNetV3Large075(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileNetV3Large(**locals(), width_ratio=0.75, model_name="mobilenetv3_large_075", **kwargs)


def MobileNetV3Large100(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileNetV3Large(**locals(), model_name="mobilenetv3_large_100", **kwargs)


def MobileNetV3Small050(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    fix_stem = True
    return MobileNetV3Small(**locals(), width_ratio=0.5, model_name="mobilenetv3_small_050", **kwargs)


def MobileNetV3Small075(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileNetV3Small(**locals(), width_ratio=0.75, model_name="mobilenetv3_small_075", **kwargs)


def MobileNetV3Small100(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileNetV3Small(**locals(), model_name="mobilenetv3_small_100", **kwargs)
