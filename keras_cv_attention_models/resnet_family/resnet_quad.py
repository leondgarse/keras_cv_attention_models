from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, drop_block, quad_stem, add_pre_post_process


PRETRAINED_DICT = {
    "resnet51q": {"imagenet": "2b54c5e252bd58f37454e6fb273716f7"},
}


def quad_block(inputs, filters, groups_div=32, strides=1, conv_shortcut=False, expansion=4, extra_conv=False, drop_rate=0, activation="swish", name=""):
    expanded_filter = filters * expansion
    groups = filters // groups_div if groups_div != 0 else 1
    if conv_shortcut:
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shortcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    if groups != 1:  # Edge block
        nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    else:
        nn = inputs

    nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", groups=groups, name=name + "groups_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "2_")

    if extra_conv:
        nn = conv2d_no_bias(nn, filters, 3, strides=1, padding="SAME", groups=groups, name=name + "extra_groups_")
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "extra_2_")

    nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "3_")

    # print(">>>> shortcut:", shortcut.shape, "nn:", nn.shape)
    nn = drop_block(nn, drop_rate)
    nn = layers.Add(name=name + "add")([shortcut, nn])
    return layers.Activation(activation, name=name + "output")(nn)


def quad_stack(inputs, blocks, filters, groups_div, strides=2, expansion=4, extra_conv=False, stack_drop=0, activation="swish", name=""):
    nn = inputs
    stack_drop_s, stack_drop_e = stack_drop if isinstance(stack_drop, (list, tuple)) else [stack_drop, stack_drop]
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * expansion) else False
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        block_drop_rate = stack_drop_s + (stack_drop_e - stack_drop_s) * id / blocks
        nn = quad_block(nn, filters, groups_div, cur_strides, conv_shortcut, expansion, extra_conv, block_drop_rate, activation, name=block_name)
    return nn


def ResNetQ(
    num_blocks,
    out_channels=[64, 128, 384, 384],
    stem_width=128,
    stem_act=False,
    expansion=4,
    groups_div=32,
    extra_conv=False,
    num_features=2048,
    strides=2,
    stem_downsample=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="resnetq",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(shape=input_shape)
    nn = quad_stem(inputs, stem_width, activation=activation, stem_act=stem_act, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_downsample:
        nn = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    total_blocks = sum(num_blocks)
    global_block_id = 0
    drop_connect_s, drop_connect_e = 0, drop_connect_rate
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        stack_drop_s = drop_connect_rate * global_block_id / total_blocks
        stack_drop_e = drop_connect_rate * (global_block_id + num_block) / total_blocks
        stack_drop = (stack_drop_s, stack_drop_e)
        cur_expansion = expansion[id] if isinstance(expansion, (list, tuple)) else expansion
        cur_extra_conv = extra_conv[id] if isinstance(extra_conv, (list, tuple)) else extra_conv
        cur_groups_div = groups_div[id] if isinstance(groups_div, (list, tuple)) else groups_div
        nn = quad_stack(nn, num_block, out_channel, cur_groups_div, stride, cur_expansion, cur_extra_conv, stack_drop, activation, name=name)
        global_block_id += num_block

    if num_features != 0:  # efficientnet like
        nn = conv2d_no_bias(nn, num_features, 1, strides=1, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, name="features_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def ResNet51Q(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 6, 4]
    out_channels = [64, 128, 384, 384 * 4]
    stem_width = 128
    stem_act = False
    expansion = [4, 4, 4, 1]
    groups_div = [32, 32, 32, 1]
    extra_conv = False
    num_features = 2048
    return ResNetQ(**locals(), model_name="resnet51q", **kwargs)


def ResNet61Q(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 4, 6, 4]
    out_channels = [256, 128, 384, 384 * 4]
    stem_width = 128
    stem_act = True
    expansion = [1, 4, 4, 1]
    groups_div = [0, 32, 32, 1]
    extra_conv = [False, True, True, True]
    num_features = 2048
    return ResNetQ(**locals(), model_name="resnet61q", **kwargs)
