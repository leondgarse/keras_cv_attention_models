from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.attention_layers import (
    BiasLayer,
    ChannelAffine,
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    add_pre_post_process,
)
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco import eval_func, anchors_func

PRETRAINED_DICT = {
    "yolor_csp": {"coco": "ed0aa82a07c4e65e9cd3d2e6ad2d0548"},
    "yolor_csp_x": {"coco": "615125ce1cd1c855f8045bf079456598"},
    "yolor_p6": {"coco": "059c6d0dd8ca869f843081b13f88f7f4"},
    "yolor_w6": {"coco": "a3dc1e70c5064aebfd8b52609e6ee704"},
    "yolor_e6": {"coco": "556263cf6aeea5b628c1814cd126eb21"},
    "yolor_d6": {"coco": "a55469feef931a07b419c3e1be639725"},
}


""" CSPDarknet backbone """
BATCH_NORM_EPSILON = 1e-4
BATCH_NORM_MOMENTUM = 0.97


def conv_dw_pw_block(inputs, filters, kernel_size=1, strides=1, use_depthwise_conv=False, activation="swish", name=""):
    nn = inputs
    if use_depthwise_conv:
        nn = depthwise_conv2d_no_bias(nn, kernel_size, strides, padding="SAME", name=name)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "dw_")
        kernel_size, strides = 1, 1
    nn = conv2d_no_bias(nn, filters, kernel_size, strides, padding="SAME", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)
    return nn


def csp_block(inputs, expansion=0.5, use_shortcut=True, activation="swish", name=""):
    input_channels = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = conv_dw_pw_block(inputs, int(input_channels * expansion), activation=activation, name=name + "1_")
    nn = conv_dw_pw_block(nn, input_channels, kernel_size=3, strides=1, activation=activation, name=name + "2_")
    if use_shortcut:
        nn = layers.Add()([inputs, nn])
    return nn


def csp_stack(
    inputs, depth, out_channels=-1, expansion=0.5, use_shortcut=True, use_pre=False, use_post=True, use_shortcut_bn=True, activation="swish", name=""
):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    out_channels = inputs.shape[channel_axis] if out_channels == -1 else out_channels
    hidden_channels = int(out_channels * expansion)
    if use_pre:
        inputs = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_")
    if use_shortcut_bn:
        short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")
    else:
        short = conv2d_no_bias(inputs, hidden_channels, 1, name=name + "short_")

    deep = inputs if use_pre else conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_pre_")
    for id in range(depth):
        block_name = name + "block{}_".format(id + 1)
        deep = csp_block(deep, 1, use_shortcut=use_shortcut, activation=activation, name=block_name)
    if use_post:
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_post_")

    out = functional.concat([deep, short], axis=channel_axis)
    if not use_shortcut_bn:
        out = batchnorm_with_activation(out, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "concat_")

    out = conv_dw_pw_block(out, out_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def res_spatial_pyramid_pooling(inputs, depth, expansion=0.5, pool_sizes=(5, 9, 13), use_shortcut_bn=True, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels * expansion)
    if use_shortcut_bn:
        short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")
    else:
        short = conv2d_no_bias(inputs, hidden_channels, 1, name=name + "short_")

    deep = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_1_")
    deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_2_")
    deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_3_")
    pp = [layers.MaxPool2D(pool_size=ii, strides=1, padding="SAME")(deep) for ii in pool_sizes]
    deep = functional.concat([deep, *pp][::-1], axis=channel_axis)  # yolor_csp.cfg, SSP concat layers=-1,-3,-5,-6
    for id in range(depth - 1):  # First one is `pre`
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "post_{}_".format(id * 2 + 1))
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "post_{}_".format(id * 2 + 2))

    out = functional.concat([deep, short], axis=channel_axis)
    if not use_shortcut_bn:
        out = batchnorm_with_activation(out, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "concat_")
    out = conv_dw_pw_block(out, hidden_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def focus_stem(inputs, filters, kernel_size=3, strides=1, padding="valid", activation="swish", name=""):
    is_channels_last = image_data_format() == "channels_last"
    channel_axis = -1 if is_channels_last else 1
    if padding.lower() == "same":  # Handling odd input_shape
        inputs = functional.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]] if is_channels_last else [[0, 0], [0, 0], [0, 1], [0, 1]])
        patch_top_left = inputs[:, :-1:2, :-1:2] if is_channels_last else inputs[:, :, :-1:2, :-1:2]
        patch_top_right = inputs[:, :-1:2, 1::2] if is_channels_last else inputs[:, :, :-1:2, 1::2]
        patch_bottom_left = inputs[:, 1::2, :-1:2] if image_data_format() == "channels_last" else inputs[:, :, 1::2, :-1:2]
        patch_bottom_right = inputs[:, 1::2, 1::2] if image_data_format() == "channels_last" else inputs[:, :, 1::2, 1::2]
    else:
        patch_top_left = inputs[:, ::2, ::2] if is_channels_last else inputs[:, :, ::2, ::2]
        patch_top_right = inputs[:, ::2, 1::2] if is_channels_last else inputs[:, :, ::2, 1::2]
        patch_bottom_left = inputs[:, 1::2, ::2] if is_channels_last else inputs[:, :, 1::2, ::2]
        patch_bottom_right = inputs[:, 1::2, 1::2] if is_channels_last else inputs[:, :, 1::2, 1::2]
    nn = functional.concat([patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right], axis=channel_axis)
    nn = conv_dw_pw_block(nn, filters, kernel_size=kernel_size, strides=strides, activation=activation, name=name)
    return nn


def csp_conv_downsample(inputs, filters, strides=2, activation="swish", name=""):
    # DownC: https://github.com/WongKinYiu/yolor/blob/paper/models/common.py#L584
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    max_down = layers.MaxPool2D(strides, strides=strides, padding="SAME")(inputs)
    max_down = conv_dw_pw_block(max_down, filters // 2, activation=activation, name=name + "max_down_")
    conv_down = conv_dw_pw_block(inputs, inputs.shape[channel_axis], activation=activation, name=name + "conv_down_1_")
    conv_down = conv_dw_pw_block(conv_down, filters // 2, kernel_size=3, strides=strides, activation=activation, name=name + "conv_down_2_")
    return functional.concat([conv_down, max_down], axis=channel_axis)


def CSPDarknet(
    depthes=[2, 8, 8, 4],
    channels=[128, 256, 512, 1024],
    stem_width=-1,  # -1 means using channels[0] // 2
    use_focus_stem=False,
    ssp_depth=2,
    out_features=[-3, -2, -1],
    use_csp_downsample=False,
    use_shortcut_bn=True,
    use_pre=False,
    use_post=True,
    input_shape=(512, 512, 3),
    activation="swish",
    model_name="",
):
    # base_channels = int(width_mul * 64)
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))

    """ Stem """
    stem_width = stem_width if stem_width > 0 else channels[0] // 2
    if use_focus_stem:
        nn = focus_stem(inputs, stem_width, activation=activation, name="stem_")
    else:
        nn = conv_dw_pw_block(inputs, 32, kernel_size=3, strides=1, activation=activation, name="stem_1_")  # Fixed as 32
        nn = conv_dw_pw_block(nn, stem_width, kernel_size=3, strides=2, activation=activation, name="stem_2_")
        nn = csp_block(nn, expansion=0.5, activation=activation, name="stem_3_")
    features = [nn]

    """ dark blocks """
    # depthes = [max(round(depth_mul * ii), 1) for ii in [2, 8, 8, 4]]  # YOLOR_CSP depth
    # channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
    # use_spps = [False] * (len(depthes) - 1) + [True]
    for id, (channel, depth) in enumerate(zip(channels, depthes)):
        stack_name = "stack{}_".format(id + 1)
        if use_csp_downsample:
            nn = csp_conv_downsample(nn, channel, activation=activation, name=stack_name)
        else:
            nn = conv_dw_pw_block(nn, channel, 3, strides=2, activation=activation, name=stack_name + "downsample_")
        nn = csp_stack(nn, depth, use_pre=use_pre, use_post=use_post, use_shortcut_bn=use_shortcut_bn, activation=activation, name=stack_name)
        if id == len(depthes) - 1:
            # ssp_depth = max(round(depth_mul * 2), 1)
            nn = res_spatial_pyramid_pooling(nn, ssp_depth, use_shortcut_bn=use_shortcut_bn, activation=activation, name=stack_name + "spp_")
        features.append(nn)

    nn = [features[ii] for ii in out_features]
    model = models.Model(inputs, nn, name=model_name)
    return model


""" path aggregation fpn """


def upsample_merge(inputs, csp_depth, use_shortcut_bn=True, activation="swish", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    upsample = conv_dw_pw_block(inputs[-1], inputs[0].shape[channel_axis], activation=activation, name=name + "up_")

    # inputs[0] = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=name + "up")(fpn_out)
    size = functional.shape(inputs[0])[1:-1] if image_data_format() == "channels_last" else functional.shape(inputs[0])[2:]
    inputs[-1] = functional.resize(upsample, size, method="nearest")
    nn = functional.concat(inputs, axis=channel_axis)
    use_shortcut, use_pre, use_post = False, True, False
    nn = csp_stack(nn, csp_depth, nn.shape[channel_axis] // 2, 1.0, use_shortcut, use_pre, use_post, use_shortcut_bn, activation=activation, name=name)
    return nn


def downsample_merge(inputs, csp_depth, use_csp_downsample=False, use_shortcut_bn=True, activation="swish", name=""):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    if use_csp_downsample:
        inputs[0] = csp_conv_downsample(inputs[0], inputs[-1].shape[channel_axis], activation=activation, name=name)
    else:
        inputs[0] = conv_dw_pw_block(inputs[0], inputs[-1].shape[channel_axis], 3, 2, activation=activation, name=name + "down_")
    nn = functional.concat(inputs, axis=channel_axis)
    use_shortcut, use_pre, use_post = False, True, False
    nn = csp_stack(nn, csp_depth, nn.shape[channel_axis] // 2, 1.0, use_shortcut, use_pre, use_post, use_shortcut_bn, activation=activation, name=name)
    return nn


def path_aggregation_fpn(features, fpn_depth=2, use_csp_downsample=False, use_shortcut_bn=True, activation="swish", name=""):
    # p5 ---┬-----------------┬-> out2
    #       ↓ [up -> concat]  ↑ [down -> concat]
    # p4 -> p4p5 ------------> out1
    #       ↓ [up -> concat]  ↑ [down -> concat]
    # p3 -> p3p4p5 ------------┴-> out0
    # features: [p3, p4, p5]
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    upsamples = [features[-1]]
    p_name = "p{}_".format(len(features) + 2)
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, ii in enumerate(features[:-1][::-1]):
        cur_p_name = "p{}".format(len(features) + 1 - id)
        nn = conv_dw_pw_block(ii, ii.shape[channel_axis] // 2, kernel_size=1, activation=activation, name=name + cur_p_name + "_down_")
        p_name = cur_p_name + p_name
        nn = upsample_merge([nn, upsamples[-1]], fpn_depth, use_shortcut_bn, activation=activation, name=name + p_name)
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        nn = downsample_merge([downsamples[-1], ii], fpn_depth, use_csp_downsample, use_shortcut_bn, activation=activation, name=cur_name)
        downsamples.append(nn)
    return downsamples


""" YOLORHead """


def yolor_head_single(inputs, filters, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    initializer = initializers.truncated_normal(stddev=0.2)

    nn = conv_dw_pw_block(inputs, filters, 3, activation=activation, name=name + "1_")
    nn = BiasLayer(initializer=initializer, axis=channel_axis, name=name + "shift_channel")(nn)

    ouput_classes = num_classes + (5 if use_object_scores else 4)  # num_anchors = 3, num_anchors * (80 + 5) = 255
    nn = layers.Conv2D(ouput_classes * num_anchors, kernel_size=1, name=name + "2_conv")(nn)
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)
    control_channels_layer = ChannelAffine(use_bias=False, axis=-1, name=name + "control_channel")
    control_channels_layer.ww_init = initializer
    nn = control_channels_layer(nn)
    # return nn
    return layers.Reshape([-1, ouput_classes], name=name + "output_reshape")(nn)


def yolor_head(inputs, num_classes=80, num_anchors=1, use_object_scores=True, activation="swish", classifier_activation="sigmoid", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    outputs = []
    for id, input in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        filters = int(input.shape[channel_axis] * 2)
        out = yolor_head_single(input, filters, num_classes, num_anchors, use_object_scores, activation=activation, name=cur_name)
        outputs.append(out)
    # return outputs
    outputs = functional.concat(outputs, axis=1)
    return activation_by_name(outputs, classifier_activation, name="classifier_")


""" YOLOR models """


def YOLOR(
    backbone=None,
    csp_depthes=[2, 8, 8, 4],
    csp_channels=[128, 256, 512, 1024],
    stem_width=-1,  # -1 means using csp_channels[0] // 2
    use_focus_stem=False,
    ssp_depth=2,
    csp_use_pre=False,
    csp_use_post=True,
    use_csp_downsample=False,
    use_shortcut_bn=True,
    fpn_depth=2,
    features_pick=[-3, -2, -1],
    anchors_mode="yolor",
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    input_shape=(640, 640, 3),
    num_classes=80,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolor",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
    kwargs=None,  # Not using, recieving parameter
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)

    if backbone is None:
        # Save line width...
        csp_kwargs = {"use_pre": csp_use_pre, "use_post": csp_use_post, "input_shape": input_shape, "activation": activation, "model_name": "darknet"}
        backbone = CSPDarknet(
            csp_depthes, csp_channels, stem_width, use_focus_stem, ssp_depth, features_pick, use_csp_downsample, use_shortcut_bn, **csp_kwargs
        )
        features = backbone.outputs
    else:
        if isinstance(features_pick[0], str):
            features = [backbone.get_layer(layer_name) for layer_name in features_pick]
        else:
            features = model_surgery.get_pyramide_feature_layers(backbone)
            features = [features[id] for id in features_pick]
        feature_names, features = model_surgery.align_pyramide_feature_output_by_image_data_format(features)
        print(">>>> features:", {ii: jj.shape for ii, jj in zip(feature_names, features)})

    backbone.trainable = False if freeze_backbone else True
    use_object_scores, num_anchors, anchor_scale = anchors_func.get_anchors_mode_parameters(anchors_mode, use_object_scores, num_anchors, anchor_scale)
    inputs = backbone.inputs[0]

    fpn_features = path_aggregation_fpn(features, fpn_depth, use_csp_downsample, use_shortcut_bn, activation=activation, name="pafpn_")
    outputs = yolor_head(fpn_features, num_classes, num_anchors, use_object_scores, activation, classifier_activation, name="head_")
    outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolor", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = eval_func.DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale)
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


def YOLOR_CSP(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [2, 8, 8, 4]
    csp_channels = [128, 256, 512, 1024]
    fpn_depth = 2
    ssp_depth = 2
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_csp"), **kwargs)


def YOLOR_CSPX(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [3, 10, 10, 5]
    csp_channels = [160, 320, 640, 1280]
    fpn_depth = 3
    ssp_depth = 3
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_csp_x"), **kwargs)


def YOLOR_P6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [3, 7, 7, 3, 3]
    csp_channels = [128, 256, 384, 512, 640]
    features_pick = [-4, -3, -2, -1]
    fpn_depth = 3
    ssp_depth = 2
    use_focus_stem = True
    csp_use_post = False
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_p6"), **kwargs)


def YOLOR_W6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [3, 7, 7, 3, 3]
    csp_channels = [128, 256, 512, 768, 1024]
    features_pick = [-4, -3, -2, -1]
    fpn_depth = 3
    ssp_depth = 2
    use_focus_stem = True
    csp_use_post = False
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_w6"), **kwargs)


def YOLOR_E6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [3, 7, 7, 3, 3]
    csp_channels = [160, 320, 640, 960, 1280]
    features_pick = [-4, -3, -2, -1]
    fpn_depth = 3
    ssp_depth = 2
    use_focus_stem = True
    csp_use_post = False
    use_csp_downsample = True
    use_shortcut_bn = False
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_e6"), **kwargs)


def YOLOR_D6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_depthes = [3, 15, 15, 7, 7]
    csp_channels = [160, 320, 640, 960, 1280]
    features_pick = [-4, -3, -2, -1]
    fpn_depth = 3
    ssp_depth = 2
    use_focus_stem = True
    csp_use_post = False
    use_csp_downsample = True
    use_shortcut_bn = False
    return YOLOR(**locals(), model_name=kwargs.pop("model_name", "yolor_d6"), **kwargs)
