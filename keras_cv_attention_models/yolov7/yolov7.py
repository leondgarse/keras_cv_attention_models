from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    add_pre_post_process,
)
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco import eval_func, anchors_func

PRETRAINED_DICT = {
    "yolov7_csp": {"coco": "71a00866fce212fa5fb11a6aeccc31e2"},
    "yolov7_d6": {"coco": "f1d59775cd5fcfc687b468c559bc5678"},
    "yolov7_e6": {"coco": "46d1671a9aa074c918c4509d29c16dd4"},
    "yolov7_e6e": {"coco": "b1316a8a431b34d39fc82d5591246713"},
    "yolov7_tiny": {"coco": "62e082bb032b7836d8494070f9dcacd6"},
    "yolov7_w6": {"coco": "ea153e7121645a55968a80dada90e67c"},
    "yolov7_x": {"coco": "0749c8c7d2554965b6559226aedbe4de"},
}


""" Yolov7Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(inputs, output_channel, kernel_size=1, strides=1, activation="swish", name=""):
    # print(f">>>> {inputs.shape = }, {output_channel = }, {kernel_size = }, {strides = }")
    nn = conv2d_no_bias(inputs, output_channel, kernel_size, strides, padding="SAME", name=name)
    return batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)


def __concat_stack__(inputs, filters, concats=[-1, -3, -5, -6], depth=6, mid_ratio=1.0, out_channels=-1, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    first = conv_bn(inputs, filters, kernel_size=1, strides=1, activation=activation, name=name + "1_")
    second = conv_bn(inputs, filters, kernel_size=1, strides=1, activation=activation, name=name + "2_")

    gathered = [first, second]
    mid_filters = int(mid_ratio * filters)
    for id in range(depth - 2):
        nn = conv_bn(gathered[-1], mid_filters, kernel_size=3, strides=1, activation=activation, name=name + "{}_".format(id + 3))
        gathered.append(nn)
    nn = functional.concat([gathered[ii] for ii in concats], axis=channel_axis)
    out_channels = out_channels if out_channels > 0 else nn.shape[channel_axis]
    nn = conv_bn(nn, out_channels, kernel_size=1, strides=1, activation=activation, name=name + "out_")
    return nn


def concat_stack(inputs, filters, concats=None, depth=6, mid_ratio=1.0, out_channels=-1, use_additional_stack=False, activation="swish", name=""):
    concats = concats if concats is not None else [-(ii + 1) for ii in range(depth)]  # [-1, -2, -3, -4, -5, -6] if None and depth=6
    nn = __concat_stack__(inputs, filters, concats, depth=depth, mid_ratio=mid_ratio, out_channels=out_channels, activation=activation, name=name)
    if use_additional_stack:
        cur_name = name + "another_"
        parallel = __concat_stack__(inputs, filters, concats, depth=depth, mid_ratio=mid_ratio, out_channels=out_channels, activation=activation, name=cur_name)
        nn = layers.Add()([nn, parallel])
    return nn


def csp_downsample(inputs, ratio=0.5, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    hidden_ratio, out_ratio = ratio if isinstance(ratio, (list, tuple)) else (ratio, ratio)
    hidden_channel, out_channel = int(input_channel * hidden_ratio), int(input_channel * out_ratio)
    pool_branch = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name=name + "pool")(inputs)
    if out_channel == 0:
        nn = pool_branch  # Maxpool only
    else:
        pool_branch = conv_bn(pool_branch, out_channel, kernel_size=1, strides=1, activation=activation, name=name + "pool_")
        conv_branch = conv_bn(inputs, hidden_channel, kernel_size=1, strides=1, activation=activation, name=name + "conv_1_")
        conv_branch = conv_bn(conv_branch, out_channel, kernel_size=3, strides=2, activation=activation, name=name + "conv_2_")

        nn = functional.concat([conv_branch, pool_branch], axis=channel_axis)
    return nn


# Almost same with yolor, just supporting YOLOV7_Tiny with depth=1
def res_spatial_pyramid_pooling(inputs, depth=2, expansion=0.5, pool_sizes=(5, 9, 13), activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels * expansion)
    short = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_1_")
    if depth > 1:  # depth = 1 for yolov7_tiny
        deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_2_")
        deep = conv_bn(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_3_")
    pp = [layers.MaxPool2D(pool_size=ii, strides=1, padding="SAME")(deep) for ii in pool_sizes]
    deep = functional.concat([deep, *pp], axis=channel_axis)  # yolov7 SPPCSPC concat, different from yolor
    for id in range(depth - 1):  # First one is `pre`
        deep = conv_bn(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "post_{}_".format(id * 2 + 1))
        deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "post_{}_".format(id * 2 + 2))

    if depth == 1:  # For yolov7_tiny
        deep = conv_bn(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "post_1_")

    out = functional.concat([deep, short], axis=channel_axis)
    out = conv_bn(out, hidden_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


# Same with yolor
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
    nn = conv_bn(nn, filters, kernel_size=kernel_size, strides=strides, activation=activation, name=name)
    return nn


def YOLOV7Backbone(
    channels=[64, 128, 256, 256],
    stack_concats=[-1, -3, -5, -6],
    stack_depth=6,
    stack_out_ratio=1.0,
    use_additional_stack=False,
    stem_width=-1,  # -1 means using channels[0]
    stem_type="conv3",  # One of ["conv3", "focus", "conv1"], "focus" for YOLOV7_*6 models, "conv1" for YOLOV7_Tiny
    csp_downsample_ratios=[0, 0.5, 0.5, 0.5],
    out_features=[-3, -2, -1],
    spp_depth=2,
    input_shape=(512, 512, 3),
    activation="swish",
    model_name="yolov7_backbone",
):
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    channel_axis = -1 if image_data_format() == "channels_last" else 1

    """ Stem """
    stem_width = stem_width if stem_width > 0 else channels[0]
    if stem_type == "focus":
        nn = focus_stem(inputs, stem_width, activation=activation, name="stem_")
    elif stem_type == "conv1":
        nn = conv_bn(inputs, stem_width, kernel_size=3, strides=2, activation=activation, name="stem_")
    else:
        nn = conv_bn(inputs, stem_width // 2, kernel_size=3, strides=1, activation=activation, name="stem_1_")
        nn = conv_bn(nn, stem_width, kernel_size=3, strides=2, activation=activation, name="stem_2_")
        nn = conv_bn(nn, stem_width, kernel_size=3, strides=1, activation=activation, name="stem_3_")

    common_kwargs = {
        "concats": stack_concats,
        "depth": stack_depth,
        "mid_ratio": 1.0,
        "use_additional_stack": use_additional_stack,
        "activation": activation,
    }

    """ blocks """
    features = [nn]
    for id, (channel, csp_downsample_ratio) in enumerate(zip(channels, csp_downsample_ratios)):
        stack_name = "stack{}_".format(id + 1)
        if isinstance(csp_downsample_ratio, (list, tuple)) or 0 < csp_downsample_ratio <= 1:
            nn = csp_downsample(nn, ratio=csp_downsample_ratio, activation=activation, name=stack_name + "downsample_")
        else:
            # nn = conv_bn(nn, nn.shape[channel_axis] * 2, kernel_size=3, strides=2, activation=activation, name=stack_name + "downsample_")
            ds_channels = nn.shape[channel_axis] * 2 if csp_downsample_ratio <= 0 else csp_downsample_ratio
            nn = conv_bn(nn, ds_channels, kernel_size=3, strides=2, activation=activation, name=stack_name + "downsample_")
        out_channels = -1 if stack_out_ratio == 1 else int(channel * len(stack_concats) * stack_out_ratio)
        nn = concat_stack(nn, channel, **common_kwargs, out_channels=out_channels, name=stack_name)

        if id == len(channels) - 1:
            # add SPPCSPC block if it's the last stack
            nn = res_spatial_pyramid_pooling(nn, depth=spp_depth, activation=activation, name=stack_name + "spp_")
        features.append(nn)

    nn = [features[ii] for ii in out_features]
    model = models.Model(inputs, nn, name=model_name)
    return model


""" path aggregation fpn, using `concat_stack` instead of `csp_stack` from yolor """


def upsample_merge(inputs, hidden_channels, mid_ratio=0.5, concats=None, depth=6, use_additional_stack=False, activation="swish", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    upsample = conv_bn(inputs[-1], inputs[0].shape[channel_axis], activation=activation, name=name + "up_")

    # inputs[0] = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=name + "up")(fpn_out)
    size = functional.shape(inputs[0])[1:-1] if image_data_format() == "channels_last" else functional.shape(inputs[0])[2:]
    inputs[-1] = functional.resize(upsample, size, method="nearest")
    nn = functional.concat(inputs, axis=channel_axis)
    out_channels = nn.shape[channel_axis] // 2
    hidden_channels = hidden_channels if hidden_channels > 0 else nn.shape[channel_axis] // 2
    nn = concat_stack(nn, hidden_channels, concats, depth, mid_ratio, out_channels, use_additional_stack, activation=activation, name=name)
    return nn


def downsample_merge(
    inputs, hidden_channels, mid_ratio=0.5, concats=None, depth=6, csp_downsample_ratio=1, use_additional_stack=False, activation="swish", name=""
):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    if isinstance(csp_downsample_ratio, (list, tuple)) or csp_downsample_ratio > 0:
        inputs[0] = csp_downsample(inputs[0], ratio=csp_downsample_ratio, activation=activation, name=name)
    else:
        inputs[0] = conv_bn(inputs[0], inputs[-1].shape[channel_axis], kernel_size=3, strides=2, activation=activation, name=name)

    nn = functional.concat(inputs, axis=channel_axis)
    out_channels = nn.shape[channel_axis] // 2
    hidden_channels = hidden_channels if hidden_channels > 0 else nn.shape[channel_axis] // 2
    nn = concat_stack(nn, hidden_channels, concats, depth, mid_ratio, out_channels, use_additional_stack, activation=activation, name=name)
    return nn


def path_aggregation_fpn(
    features, hidden, mid_ratio=0.5, channel_ratio=0.25, concats=None, depth=6, csp_downsample_ratio=1, use_additional_stack=False, activation="swish", name=""
):
    # yolov7                                                        # yolov7_w6
    # 51: p5 512 ---+---------------------+-> 101: out2 512         # 47: p5 512 ---┬---------------------┬-> 113: out 512
    #               v [up 256 -> concat]  ^ [down 512 -> concat]    #               ↓ [up 384 -> concat]  ↑[down 512 -> concat]
    # 37: p4 1024 -> 63: p4p5 256 -------> 88: out1 256             # 37: p4 768 --- 59: p4p5 384 ------- 103: out 384
    #               v [up 128 -> concat]  ^ [down 256 -> concat]    #               ↓ [up 256 -> concat]  ↑[down 384 -> concat]
    # 24: p3 512 --> 75: p3p4p5 128 ------+--> 75: out0 128         # 28: p3 512 --- 71: p3p4p5 256 -- 93: out 256
    #                                                               #               ↓ [up 128 -> concat]  ↑[down 256 -> concat]
    #                                                               # 19: p2 256 --- 83: p2p3p4p5 128 -----┴-> 83: out 128
    # features: [p3, p4, p5]
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    hidden_channels = hidden.copy() if isinstance(hidden, list) else hidden
    upsamples = [features[-1]]
    p_name = "p{}_".format(len(features) + 2)
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, ii in enumerate(features[:-1][::-1]):
        cur_p_name = "p{}".format(len(features) + 1 - id)
        nn = conv_bn(ii, int(ii.shape[channel_axis] * channel_ratio), kernel_size=1, activation=activation, name=name + cur_p_name + "_down_")
        hidden_channel = hidden_channels.pop(0) if isinstance(hidden_channels, list) else hidden_channels
        p_name = cur_p_name + p_name
        nn = upsample_merge([nn, upsamples[-1]], hidden_channel, mid_ratio, concats, depth, use_additional_stack, activation=activation, name=name + p_name)
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        hidden_channel = hidden_channels.pop(0) if isinstance(hidden_channels, list) else hidden_channels
        cur_csp_downsample_ratio = csp_downsample_ratio.pop(0) if isinstance(csp_downsample_ratio, list) else csp_downsample_ratio
        nn = downsample_merge(
            [downsamples[-1], ii], hidden_channel, mid_ratio, concats, depth, cur_csp_downsample_ratio, use_additional_stack, activation, name=cur_name
        )
        downsamples.append(nn)
    return downsamples


""" YOLOV7Head, using Reparam Conv block """


def yolov7_head_single(inputs, filters, use_reparam_conv_head=True, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", name=""):
    if use_reparam_conv_head:
        # OREPA_3x3_RepConv
        rep_conv_3 = conv_bn(inputs, filters, 3, activation=None, name=name + "3x3_")
        rep_conv_1 = conv_bn(inputs, filters, 1, activation=None, name=name + "1x1_")
        nn = layers.Add()([rep_conv_3, rep_conv_1])
        nn = activation_by_name(nn, activation=activation, name=name)
    else:
        nn = conv_bn(inputs, filters, 3, activation=activation, name=name + "1_")

    ouput_classes = num_classes + (5 if use_object_scores else 4)  # num_anchors = 3, num_anchors * (80 + 5) = 255
    nn = layers.Conv2D(ouput_classes * num_anchors, kernel_size=1, name=name + "2_conv")(nn)
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)
    # return nn
    return layers.Reshape([-1, ouput_classes], name=name + "output_reshape")(nn)


def yolov7_head(
    inputs, use_reparam_conv_head=True, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", classifier_activation="sigmoid", name=""
):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    outputs = []
    for id, input in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        filters = int(input.shape[channel_axis] * 2)
        out = yolov7_head_single(input, filters, use_reparam_conv_head, num_classes, num_anchors, use_object_scores, activation=activation, name=cur_name)
        outputs.append(out)
    # return outputs
    outputs = functional.concat(outputs, axis=1)
    return activation_by_name(outputs, classifier_activation, name="classifier_")


""" YOLOV7 models, almost same with yolor """


def YOLOV7(
    backbone=None,
    csp_channels=[64, 128, 256, 256],  # [YOLOV7Backbone parameters]
    stack_concats=[-1, -3, -5, -6],
    stack_depth=6,
    stack_out_ratio=1.0,
    use_additional_stack=False,
    stem_width=-1,  # -1 means using csp_channels[0] // 2
    stem_type="conv3",  # One of ["conv3", "focus", "conv1"], "focus" for YOLOV7_*6 models, "conv1" for YOLOV7_Tiny
    csp_downsample_ratios=[0, 0.5, 0.5, 0.5],
    spp_depth=2,
    fpn_hidden_channels=[256, 128, 256, 512],  # [FPN parameters]
    fpn_channel_ratio=0.25,
    fpn_stack_concats=None,
    fpn_stack_depth=-1,  # -1 for using same with stack_depth
    fpn_mid_ratio=0.5,
    fpn_csp_downsample_ratio=1,
    use_reparam_conv_head=True,
    features_pick=[-3, -2, -1],  # [Detector parameters]
    anchors_mode="yolor",
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    input_shape=(640, 640, 3),
    num_classes=80,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolov7",
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
        csp_kwargs = {"out_features": features_pick, "spp_depth": spp_depth, "input_shape": input_shape, "activation": activation, "model_name": "backbone"}
        backbone = YOLOV7Backbone(
            csp_channels, stack_concats, stack_depth, stack_out_ratio, use_additional_stack, stem_width, stem_type, csp_downsample_ratios, **csp_kwargs
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

    # Save line width...
    fpn_stack_depth = fpn_stack_depth if fpn_stack_depth > 0 else stack_depth
    fpn_kwargs = {"csp_downsample_ratio": fpn_csp_downsample_ratio, "use_additional_stack": use_additional_stack, "activation": activation, "name": "pafpn_"}
    fpn_features = path_aggregation_fpn(features, fpn_hidden_channels, fpn_mid_ratio, fpn_channel_ratio, fpn_stack_concats, fpn_stack_depth, **fpn_kwargs)

    outputs = yolov7_head(fpn_features, use_reparam_conv_head, num_classes, num_anchors, use_object_scores, activation, classifier_activation, name="head_")
    outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolov7", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = eval_func.DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale)
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


def YOLOV7_Tiny(
    input_shape=(416, 416, 3),
    freeze_backbone=False,
    num_classes=80,
    backbone=None,
    activation="leaky_relu/0.1",
    classifier_activation="sigmoid",
    pretrained="coco",
    **kwargs,
):
    # anchors_yolov7_tiny = np.array([[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]])
    # anchors_yolor = np.array([[12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]])
    # anchors_yolov7_tiny == np.ceil((anchors_yolor * 416 / 512)).astype('int') [TODO]
    stem_type = "conv1"
    csp_channels = [32, 64, 128, 256]
    stack_concats = [-1, -2, -3, -4]
    stack_depth = 4
    stack_out_ratio = 0.5
    csp_downsample_ratios = [0, [0, 0], [0, 0], [0, 0]]  # First 0 for conv_bn downsmaple, others [0, 0] means maxpool
    spp_depth = 1

    fpn_hidden_channels = [64, 32, 64, 128]
    fpn_mid_ratio = 1.0
    fpn_channel_ratio = 0.5
    fpn_csp_downsample_ratio = [0, 0]  # [0, 0] means using conv_bn downsmaple
    use_reparam_conv_head = False
    return YOLOV7(**locals(), model_name=kwargs.pop("model_name", "yolov7_tiny"), **kwargs)


def YOLOV7_CSP(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOV7(**locals(), model_name=kwargs.pop("model_name", "yolov7_csp"), **kwargs)


def YOLOV7_X(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    stack_concats = [-1, -3, -5, -7, -8]
    stack_depth = 8
    stem_width = 80

    fpn_stack_concats = [-1, -3, -5, -7, -8]
    fpn_mid_ratio = 1.0
    use_reparam_conv_head = False
    return YOLOV7(**locals(), model_name=kwargs.pop("model_name", "yolov7_x"), **kwargs)


def YOLOV7_W6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = kwargs.pop("csp_channels", [64, 128, 256, 384, 512])
    features_pick = kwargs.pop("features_pick", [-4, -3, -2, -1])
    stem_type = kwargs.pop("stem_type", "focus")
    csp_downsample_ratios = kwargs.pop("csp_downsample_ratios", [128, 256, 512, 768, 1024])  # > 1 value means using conv_bn instead of csp_downsample
    stack_out_ratio = kwargs.pop("stack_out_ratio", 0.5)

    fpn_hidden_channels = kwargs.pop("fpn_hidden_channels", [384, 256, 128, 256, 384, 512])
    fpn_channel_ratio = kwargs.pop("fpn_channel_ratio", 0.5)
    fpn_csp_downsample_ratio = kwargs.pop("fpn_csp_downsample_ratio", 0)
    use_reparam_conv_head = kwargs.pop("use_reparam_conv_head", False)

    kwargs.pop("kwargs", None)  # From other YOLOV7_*6 models
    return YOLOV7(**locals(), model_name=kwargs.pop("model_name", "yolov7_w6"), **kwargs)


def YOLOV7_E6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    stack_concats = [-1, -3, -5, -7, -8]
    stack_depth = 8
    stem_width = 80
    csp_downsample_ratios = [1, 1, 1, [1, 480 / 640], [1, 640 / 960]]  # different from YOLOV7_W6

    fpn_mid_ratio = 0.5
    fpn_csp_downsample_ratio = [1, [1, 240 / 320], [1, 320 / 480]]  # different from YOLOV7_W6

    kwargs.pop("kwargs", None)  # From YOLOV7_E6E / YOLOV7_D6
    return YOLOV7_W6(**locals(), model_name=kwargs.pop("model_name", "yolov7_e6"), **kwargs)


def YOLOV7_D6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    stack_concats = [-1, -3, -5, -7, -9, -10]
    stack_depth = 10
    stem_width = 96
    return YOLOV7_E6(**locals(), model_name=kwargs.pop("model_name", "yolov7_d6"), **kwargs)


def YOLOV7_E6E(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    use_additional_stack = True
    return YOLOV7_E6(**locals(), model_name=kwargs.pop("model_name", "yolov7_e6e"), **kwargs)
