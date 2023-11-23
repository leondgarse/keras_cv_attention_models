import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    batchnorm_with_activation,
    drop_block,
    make_divisible,
    se_module,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_EPSILON = 1e-5

PRETRAINED_DICT = {
    "repvit_m_09": {"imagenet": "f406182ed50349a079093717b67fba1a"},
    "repvit_m_10": {"imagenet": "1ff41e710ccc3dc46954c356ff62f20c"},
    "repvit_m_11": {"imagenet": "30c9ccc796392e1adddda630858d7646"},
    "repvit_m_15": {"imagenet": "b9347ab0ed9c15bb129acc6ba3d5833d"},
    "repvit_m_23": {"imagenet": "ffb0f7617fb27baab4bfce5a17260c19"},
}


def rep_vgg_depthwise(inputs, kernel_size=3, strides=1, deploy=False, name=""):
    if deploy:
        return depthwise_conv2d_no_bias(inputs, kernel_size=kernel_size, strides=strides, use_bias=True, padding="same", name=name + "REPARAM_1_")

    dw_1 = depthwise_conv2d_no_bias(inputs, kernel_size=kernel_size, strides=strides, padding="same", name=name + "REPARAM_1_")
    dw_1 = batchnorm_with_activation(dw_1, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "REPARAM_1_")
    dw_2 = depthwise_conv2d_no_bias(inputs, 1, use_bias=True, name=name + "REPARAM_2_")
    out = layers.Add(name=name + "REPARAM_out")([dw_1, dw_2, inputs])
    return batchnorm_with_activation(out, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "out_")
    return


def dwconv_bn_conv_bn(inputs, out_channel, kernel_size=1, strides=1, deploy=False, name=""):
    nn = depthwise_conv2d_no_bias(inputs, kernel_size, strides=strides, use_bias=deploy, padding="same", name=name + "1_")
    nn = nn if deploy else batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, name=name + "1_")
    nn = conv2d_no_bias(nn, out_channel, use_bias=deploy, name=name + "2_")
    nn = nn if deploy else batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "2_")
    return nn


def conv_bn_act_conv_bn(
    inputs, hidden_channel, out_channel, use_residual=True, kernel_size=1, strides=1, deploy=False, drop_rate=0, activation="gelu", name=""
):
    nn = conv2d_no_bias(inputs, hidden_channel, kernel_size, strides=strides, use_bias=deploy, padding="same", name=name + "1_")
    nn = nn if deploy else batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "1_")
    nn = activation_by_name(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, out_channel, kernel_size, strides=strides, use_bias=deploy, padding="same", name=name + "2_")
    nn = nn if deploy else batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "2_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return (nn + inputs) if use_residual else nn


def RepViT(
    num_blocks=[3, 3, 15, 2],
    out_channels=[48, 96, 192, 384],
    stem_width=-1,
    se_ratio=0.25,  # will use `se_module` every other block in each stack if > 0
    input_shape=(224, 224, 3),
    deploy=False,  # build model with rep_vgg_depthwise/conv+bn/distill_head all being fused
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    use_distillation=False,
    dropout=0,
    pretrained=None,
    model_name="repvit",
    kwargs=None,
):
    """Patch stem"""
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv_bn_act_conv_bn(
        inputs, stem_width // 2, stem_width, use_residual=False, kernel_size=3, strides=2, deploy=deploy, activation=activation, name="stem_"
    )

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        out_channel = make_divisible(out_channel, 8)
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = dwconv_bn_conv_bn(nn, out_channel, kernel_size=3, strides=2, deploy=deploy, name=stack_name + "downsample_token_mixer_")
            nn = conv_bn_act_conv_bn(nn, out_channel * 2, out_channel, deploy=deploy, activation=activation, name=stack_name + "downsample_channel_mixer_")

        for block_id in range(num_block):
            name = "stack{}_block{}_".format(stack_id + 1, block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            cur_se_ratio = 0 if block_id % 2 == 1 or block_id == (num_block - 1) else se_ratio
            # print(f"{name = }, {cur_se_ratio = }")
            nn = rep_vgg_depthwise(nn, deploy=deploy, name=name + "repvgg_")
            nn = se_module(nn, cur_se_ratio, activation="relu", name=name + "se_") if cur_se_ratio > 0 else nn
            nn = conv_bn_act_conv_bn(
                nn, out_channel * 2, out_channel, deploy=deploy, activation=activation, drop_rate=block_drop_rate, name=name + "channel_mixer_"
            )
            global_block_id += 1

    if deploy and num_classes > 0:
        nn = layers.GlobalAveragePooling2D()(nn)
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)
    elif num_classes > 0:
        nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)
        out = batchnorm_with_activation(nn, activation=None, name="head_")
        out = layers.Flatten()(out)
        out = layers.Dropout(dropout, name="head_drop")(out) if dropout > 0 else out
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(out)

        if use_distillation:
            distill = batchnorm_with_activation(nn, activation=None, name="distill_head_")
            distill = layers.Flatten()(distill)
            distill = layers.Dropout(dropout, name="distill_head_drop")(distill) if dropout > 0 else distill
            distill = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(distill)
            out = [out, distill]
    else:
        out = nn
    model = models.Model(inputs, out, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "repvit", pretrained)

    add_pre_post_process(model, rescale_mode="torch")
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    return model


def switch_to_deploy(model):
    from keras_cv_attention_models.model_surgery.model_surgery import fuse_reparam_blocks, convert_to_fused_conv_bn_model, fuse_distill_head

    new_model = fuse_distill_head(model) if "head" in model.output_names else model
    new_model = convert_to_fused_conv_bn_model(fuse_reparam_blocks(convert_to_fused_conv_bn_model(new_model)))
    add_pre_post_process(new_model, rescale_mode=model.preprocess_input.rescale_mode, post_process=model.decode_predictions)
    return new_model


@register_model
def RepViT_M09(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, use_distillation=False, classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    return RepViT(**locals(), model_name="repvit_m_09" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def RepViT_M10(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, use_distillation=False, classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    num_blocks = [3, 3, 15, 2]
    out_channels = [56, 112, 224, 448]
    return RepViT(**locals(), model_name="repvit_m_10" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def RepViT_M11(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, use_distillation=False, classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    num_blocks = [3, 3, 13, 2]
    out_channels = [64, 128, 256, 512]
    return RepViT(**locals(), model_name="repvit_m_11" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def RepViT_M15(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, use_distillation=False, classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    num_blocks = [5, 5, 25, 4]
    out_channels = [64, 128, 256, 512]
    return RepViT(**locals(), model_name="repvit_m_15" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def RepViT_M23(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, use_distillation=False, classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    num_blocks = [7, 7, 35, 2]
    out_channels = [80, 160, 320, 640]
    return RepViT(**locals(), model_name="repvit_m_23" + ("_deploy" if deploy else ""), **kwargs)
