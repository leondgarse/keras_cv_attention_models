from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    mlp_block,
    multi_head_self_attention,
    se_module,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "fastvit_ma36": {"imagenet": "13d53c4cfb2587962e779027efdc1104", "distill": "8a0e72c0171b59091b484849f4b7cd87"},
    "fastvit_s12": {"imagenet": "de71db346a180d81e7195291b926b393", "distill": "a0d97a2ef840ab403d12513d40807ffd"},
    "fastvit_sa12": {"imagenet": "4c3747e8915eec9359395984dd832555", "distill": "d656d94844af0e34ec57f38b363395a3"},
    "fastvit_sa24": {"imagenet": "11dd3d252e8ad205a69274d196a7bea2", "distill": "6465e5f6218735eae17a06ca27839a64"},
    "fastvit_sa36": {"imagenet": "b93fed540c0b5acc1cdd99b54173ecc0", "distill": "0ef091d7ceeecfc5a4c2d05d5ebdd297"},
    "fastvit_t12": {"imagenet": "2f6e745fbfdfcd375bd10e8e3d58663f", "distill": "2880385ec6a2225c59852a18b3be0bbb"},
    "fastvit_t8": {"imagenet": "58b3cf9b0f5072b7b0a605c60f76328d", "distill": "90147a9dd53178f5c046c6cdd83137c0"},
}


def rep_conv_block(inputs, out_channel, kernel_size=3, strides=1, groups=1, deploy=False, name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    use_depthwise = input_channel == out_channel and groups == out_channel
    if deploy and use_depthwise:
        return depthwise_conv2d_no_bias(inputs, kernel_size=kernel_size, strides=strides, use_bias=True, padding="same", name=name + "REPARAM_1_")
    elif deploy:
        return conv2d_no_bias(
            inputs, out_channel, kernel_size=kernel_size, strides=strides, use_bias=True, padding="same", groups=groups, name=name + "REPARAM_1_"
        )

    use_shortcut = input_channel == out_channel and strides == 1
    if use_shortcut:
        short = batchnorm_with_activation(inputs, epsilon=BATCH_NORM_EPSILON, name=name + "REPARAM_0_")

    if use_depthwise:
        dw_1 = depthwise_conv2d_no_bias(inputs, kernel_size=kernel_size, strides=strides, padding="same", name=name + "REPARAM_1_")
    else:
        dw_1 = conv2d_no_bias(inputs, out_channel, kernel_size=kernel_size, strides=strides, padding="same", groups=groups, name=name + "REPARAM_1_")
    dw_1 = batchnorm_with_activation(dw_1, epsilon=BATCH_NORM_EPSILON, name=name + "REPARAM_1_")

    if kernel_size > 1:
        if use_depthwise:
            dw_2 = depthwise_conv2d_no_bias(inputs, 1, strides=strides, name=name + "REPARAM_2_")
        else:
            dw_2 = conv2d_no_bias(inputs, out_channel, 1, strides=strides, groups=groups, name=name + "REPARAM_2_")
        dw_2 = batchnorm_with_activation(dw_2, epsilon=BATCH_NORM_EPSILON, name=name + "REPARAM_2_")
        out = layers.Add(name=name + "REPARAM_out")([dw_1, dw_2, short] if use_shortcut else [dw_1, dw_2])
    else:
        out = layers.Add(name=name + "REPARAM_out")([dw_1, short]) if use_shortcut else dw_1
    return out


def rep_downsample_block(inputs, out_channel, strides=2, deploy=False, name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    if deploy:
        return conv2d_no_bias(inputs, out_channel, 7, strides=strides, use_bias=True, padding="same", groups=input_channel, name=name + "REPARAM_1_")

    dw_1 = conv2d_no_bias(inputs, out_channel, kernel_size=7, strides=strides, padding="same", groups=input_channel, name=name + "REPARAM_1_")
    dw_1 = batchnorm_with_activation(dw_1, epsilon=BATCH_NORM_EPSILON, name=name + "REPARAM_1_")
    dw_2 = conv2d_no_bias(inputs, out_channel, kernel_size=3, strides=strides, padding="same", groups=input_channel, name=name + "REPARAM_2_")
    dw_2 = batchnorm_with_activation(dw_2, epsilon=BATCH_NORM_EPSILON, name=name + "REPARAM_2_")
    return layers.Add(name=name + "REPARAM_out")([dw_1, dw_2])


def rep_conditional_positional_encoding(inputs, kernel_size=7, deploy=False, name=""):
    dw_1 = depthwise_conv2d_no_bias(inputs, kernel_size=kernel_size, strides=1, use_bias=True, padding="same", name=name + "REPARAM_1_")
    return dw_1 if deploy else layers.Add(name=name + "REPARAM_out")([dw_1, inputs])


def mixer_mlp_block(inputs, out_channel, mlp_ratio=3, use_attn=False, kernel_size=3, drop_rate=0, layer_scale=-1, deploy=False, activation="gelu", name=None):
    # print(f"{inputs.shape = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    layer_scale = 0 if deploy else layer_scale  # Force skip layer_scale if deploy, as it will be fused
    norm_mixer = batchnorm_with_activation(inputs, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "mixer_")
    if use_attn:
        mixer = norm_mixer if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(norm_mixer)
        mixer = multi_head_self_attention(mixer, num_heads=input_channel // 32, qkv_bias=False, out_bias=True, name=name + "attn_")
        mixer = mixer if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(mixer)
    else:
        deep_mixer = rep_conv_block(inputs, out_channel, kernel_size=kernel_size, groups=out_channel, deploy=deploy, name=name + "mixer_")
        mixer = deep_mixer if deploy else layers.Subtract(name=name + "REPARAM_TWICE_out")([deep_mixer, norm_mixer])

    if use_attn or not deploy:
        mixer = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=channel_axis, name=name + "1_gamma")(mixer) if layer_scale > 0 else mixer
        mixer = drop_block(mixer, drop_rate=drop_rate, name=name + "1_")
        mixer = layers.Add(name=name + ("1_out" if use_attn else "REPARAM_THIRD_out"))([mixer, inputs])

    mlp = conv2d_no_bias(mixer, out_channel, kernel_size=7, strides=1, use_bias=deploy, padding="same", groups=input_channel, name=name + "mlp_pre_")
    mlp = mlp if deploy else batchnorm_with_activation(mlp, epsilon=BATCH_NORM_EPSILON, activation=None, name=name + "mlp_pre_")
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), use_conv=True, activation=activation, name=name + "mlp_")

    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=channel_axis, name=name + "2_gamma")(mlp) if layer_scale > 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "2_")
    return layers.Add(name=name + "2_output")([mlp, mixer])


def FastViT(
    num_blocks=[2, 2, 4, 2],
    out_channels=[48, 96, 192, 384],
    block_types=["conv", "conv", "conv", "conv"],
    mlp_ratio=3,
    stem_width=-1,
    layer_scale=1e-5,
    input_shape=(256, 256, 3),
    deploy=False,  # build model with rep_xxx / conv+bn all being fused
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="fastvit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = rep_conv_block(inputs, stem_width, kernel_size=3, strides=2, deploy=deploy, name="stem_1_")
    nn = activation_by_name(nn, activation=activation, name="stem_1_")
    nn = rep_conv_block(nn, stem_width, kernel_size=3, strides=2, groups=stem_width, deploy=deploy, name="stem_2_")
    nn = activation_by_name(nn, activation=activation, name="stem_2_")
    nn = rep_conv_block(nn, stem_width, kernel_size=1, strides=1, deploy=deploy, name="stem_3_")
    nn = activation_by_name(nn, activation=activation, name="stem_3_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        stack_name = "stack{}_".format(stack_id + 1)
        use_attn = False if block_type[0].lower() == "c" else True
        if stack_id > 0:
            nn = rep_downsample_block(nn, out_channel, deploy=deploy, name=stack_name + "downsample_")  # [???] activation is ignored
            nn = rep_conv_block(nn, out_channel, kernel_size=1, deploy=deploy, name=stack_name + "downsample_2_")
            nn = activation_by_name(nn, activation=activation, name=stack_name + "downsample_")

        if use_attn:
            nn = rep_conditional_positional_encoding(nn, deploy=deploy, name=stack_name + "cpe_")

        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = mixer_mlp_block(
                nn, out_channel, mlp_ratio, use_attn, drop_rate=block_drop_rate, layer_scale=layer_scale, deploy=deploy, activation=activation, name=name
            )

    if num_classes > 0:
        nn = rep_conv_block(nn, out_channels[-1] * 2, kernel_size=3, strides=1, groups=out_channels[-1], deploy=deploy, name="features_")
        nn = se_module(nn, se_ratio=0.0625, divisor=1, activation="relu", name="features_se_")
        nn = activation_by_name(nn, activation=activation, name="features_")

        nn = layers.GlobalAveragePooling2D()(nn)
        nn = layers.Dropout(dropout, name="head_drop")(nn) if dropout > 0 else nn
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)
    model = models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "fastvit", pretrained)

    add_pre_post_process(model, rescale_mode="torch")
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    return model


def switch_to_deploy(model):
    from keras_cv_attention_models.model_surgery.model_surgery import fuse_reparam_blocks, convert_to_fused_conv_bn_model, fuse_channel_affine_to_conv_dense

    new_model = convert_to_fused_conv_bn_model(model)
    new_model = fuse_reparam_blocks(new_model, output_layer_key="REPARAM_out")
    new_model = fuse_reparam_blocks(new_model, output_layer_key="REPARAM_TWICE_out")
    new_model = fuse_channel_affine_to_conv_dense(new_model)
    new_model = fuse_reparam_blocks(new_model, output_layer_key="REPARAM_THIRD_out")
    add_pre_post_process(new_model, rescale_mode=model.preprocess_input.rescale_mode, post_process=model.decode_predictions)
    return new_model


@register_model
def FastViT_T8(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    return FastViT(**locals(), model_name="fastvit_t8", **kwargs)


@register_model
def FastViT_T12(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [64, 128, 256, 512]
    return FastViT(**locals(), model_name="fastvit_t12", **kwargs)


@register_model
def FastViT_S12(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [64, 128, 256, 512]
    mlp_ratio = 4
    return FastViT(**locals(), model_name="fastvit_s12", **kwargs)


@register_model
def FastViT_SA12(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [64, 128, 256, 512]
    mlp_ratio = 4
    block_types = ["conv", "conv", "conv", "transformer"]
    return FastViT(**locals(), model_name="fastvit_sa12", **kwargs)


@register_model
def FastViT_SA24(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [4, 4, 12, 4]
    out_channels = [64, 128, 256, 512]
    mlp_ratio = 4
    block_types = ["conv", "conv", "conv", "transformer"]
    return FastViT(**locals(), model_name="fastvit_sa24", **kwargs)


@register_model
def FastViT_SA36(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [6, 6, 18, 6]
    out_channels = [64, 128, 256, 512]
    mlp_ratio = 4
    layer_scale = kwargs.pop("layer_scale", 1e-6)
    block_types = ["conv", "conv", "conv", "transformer"]
    return FastViT(**locals(), model_name="fastvit_sa36", **kwargs)


@register_model
def FastViT_MA36(input_shape=(256, 256, 3), num_classes=1000, deploy=False, activation="gelu", classifier_activation="softmax", pretrained="distill", **kwargs):
    num_blocks = [6, 6, 18, 6]
    out_channels = [76, 152, 304, 608]
    mlp_ratio = 4
    layer_scale = kwargs.pop("layer_scale", 1e-6)
    block_types = ["conv", "conv", "conv", "transformer"]
    return FastViT(**locals(), model_name="fastvit_ma36", **kwargs)
