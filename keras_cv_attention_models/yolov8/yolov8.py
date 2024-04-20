import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    add_pre_post_process,
    ZeroInitGain,
)
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco import eval_func, anchors_func

PRETRAINED_DICT = {
    "yolov8_l": {"coco": "db0fcde5d2811b33b7f5f0f400d76911"},
    "yolov8_m": {"coco": "cb8c25148bb17485776ade4cf80cc6f6"},
    "yolov8_n": {"coco": "4cb83c7e452cdcd440b75546df0b211e"},
    "yolov8_s": {"coco": "4e1ac133e2a8831845172d8491c2747a"},
    "yolov8_x": {"coco": "2be28e650bf299aeea7ee26ab765a23e"},
    "yolov8_x6": {"coco": "f51ed830ccf5efae7dc56f2ce5e20890"},
    "yolov8_l_cls": {"imagenet": "071f41125034dd15401f6c6925fc1e6f"},
    "yolov8_m_cls": {"imagenet": "35ef50aa07ff232afa08f321447e354d"},
    "yolov8_n_cls": {"imagenet": "b1cfac787589689c0f2abde6893ec140"},
    "yolov8_s_cls": {"imagenet": "2caa57e8cf67b39921c35f89cea5061c"},
    "yolov8_x_cls": {"imagenet": "2d4b8b996c24f5fde903678ee8b7cf20"},
    "yolov8_l_seg": {"coco": "01ad40cd469d9fc57cae381283e389eb"},
    "yolov8_m_seg": {"coco": "9838e71d3cbde7a8d7a89aef724cd323"},
    "yolov8_n_seg": {"coco": "c6e6b46e79dc555a015f4ff2b5aaab36"},
    "yolov8_s_seg": {"coco": "d806a3fc3740c899a4f11feafe7298b6"},
    "yolov8_x_seg": {"coco": "6ce769b0119372fee11cbfe2f3b6910b"},
}


""" Yolov8Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(inputs, output_channel, kernel_size=1, strides=1, use_bias=False, activation="swish", name=""):
    # print(f">>>> {inputs.shape = }, {output_channel = }, {kernel_size = }, {strides = }")
    nn = conv2d_no_bias(inputs, output_channel, kernel_size, strides, use_bias=use_bias, padding="same", name=name)
    return batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)


def reparam_conv_bn(inputs, output_channel, kernel_size=3, strides=1, use_bias=False, use_identity=True, activation="swish", name=""):
    branch_3x3 = conv2d_no_bias(inputs, output_channel, 3, strides, use_bias=False, padding="same", name=name + "REPARAM_k3_")
    branch_3x3 = batchnorm_with_activation(branch_3x3, activation=None, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "REPARAM_k3_")

    branch_1x1 = conv2d_no_bias(inputs, output_channel, 1, strides, use_bias=use_bias, padding="valid", name=name + "REPARAM_k1_")
    # branch_1x1 = batchnorm_with_activation(branch_1x1, activation=None, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "REPARAM_k1_")

    # out = (branch_3x3 + branch_1x1 + inputs) if use_identity else (branch_3x3 + branch_1x1)
    out = layers.Add(name=name + "REPARAM_out")([branch_3x3, branch_1x1, inputs] if use_identity else [branch_3x3, branch_1x1])
    return batchnorm_with_activation(out, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)


def csp_with_2_conv(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,
    parallel_mode=True,
    use_bias=False,
    use_alpha=False,
    use_reparam_conv=False,
    activation="swish",
    name="",
):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    # short = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")  # For YOLO_NAS
    # deep = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_")  # For YOLO_NAS
    pre = conv_bn(inputs, hidden_channels * 2, kernel_size=1, activation=activation, name=name + "pre_")
    if parallel_mode:
        short, deep = functional.split(pre, 2, axis=channel_axis)
    else:  # parallel_mode=False for YOLOV8_X6 `path_aggregation_fpn` C2 module, deep branch first
        deep, short = functional.split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        cur_short = ZeroInitGain(name=name + "short_{}_alpha".format(id))(out[-1]) if use_alpha else out[-1]
        cur_name = name + "pre_{}_".format(id)
        if use_reparam_conv:
            deep = reparam_conv_bn(deep, hidden_channels, kernel_size=3, use_bias=use_bias, activation=activation, name=cur_name + "1_")
            deep = reparam_conv_bn(deep, hidden_channels, kernel_size=3, use_bias=use_bias, activation=activation, name=cur_name + "2_")
        else:
            deep = conv_bn(deep, hidden_channels, kernel_size=3, use_bias=use_bias, activation=activation, name=cur_name + "1_")
            deep = conv_bn(deep, hidden_channels, kernel_size=3, use_bias=use_bias, activation=activation, name=cur_name + "2_")

        deep = (cur_short + deep) if shortcut else deep
        out.append(deep)
    # parallel_mode=False for YOLOV8_X6 `path_aggregation_fpn` C2 module, only concat `short` and the last `deep` one.
    out = functional.concat(out, axis=channel_axis) if parallel_mode else functional.concat([deep, short], axis=channel_axis)
    # out = functional.concat([*out[1:], out[0]], axis=channel_axis) if parallel_mode else functional.concat([deep, short], axis=channel_axis)  # For YOLO_NAS
    out = conv_bn(out, channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def spatial_pyramid_pooling_fast(inputs, pool_size=5, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    nn = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_")
    pool_1 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(nn)
    pool_2 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(pool_1)
    pool_3 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(pool_2)

    out = functional.concat([nn, pool_1, pool_2, pool_3], axis=channel_axis)
    out = conv_bn(out, input_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def YOLOV8Backbone(
    channels=[32, 64, 128, 256],
    depthes=[1, 2, 2, 1],
    out_features=[-3, -2, -1],
    csp_expansions=0.5,
    csp_parallel_mode=True,
    use_alpha=False,
    use_bias=False,
    use_reparam_conv=False,
    input_shape=(640, 640, 3),
    activation="swish",
    num_classes=0,  # > 0 value for classification model
    dropout=0,  # for classification model
    classifier_activation="softmax",  # for classification model
    pretrained=None,  # for classification model
    model_name="yolov8_backbone",
    kwargs=None,  # Not using, recieving parameter
):
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    is_classification_model = num_classes > 0

    global BATCH_NORM_EPSILON
    global BATCH_NORM_MOMENTUM
    if is_classification_model:
        BATCH_NORM_EPSILON = 1e-5
        BATCH_NORM_MOMENTUM = 0.9
    else:
        BATCH_NORM_EPSILON = 1e-3
        BATCH_NORM_MOMENTUM = 0.97

    """ Stem """
    # stem_width = stem_width if stem_width > 0 else channels[0]
    stem_width = channels[0]
    if use_reparam_conv:
        nn = reparam_conv_bn(inputs, stem_width // 2, kernel_size=3, strides=2, use_bias=use_bias, use_identity=False, activation=activation, name="stem_1_")
        nn = reparam_conv_bn(nn, stem_width, kernel_size=3, strides=2, use_bias=use_bias, use_identity=False, activation=activation, name="stem_2_")
    else:
        nn = conv_bn(inputs, stem_width // 2, kernel_size=3, strides=2, use_bias=use_bias, activation=activation, name="stem_1_")
        nn = conv_bn(nn, stem_width, kernel_size=3, strides=2, use_bias=use_bias, activation=activation, name="stem_2_")

    """ blocks """
    block_kwargs = dict(use_bias=use_bias, use_alpha=use_alpha, use_reparam_conv=use_reparam_conv, activation=activation)
    features = [nn]
    for stack_id, (channel, depth) in enumerate(zip(channels, depthes)):
        stack_name = "stack{}_".format(stack_id + 1)
        cur_name = stack_name + "downsample_"
        if stack_id >= 1:
            if use_reparam_conv:
                nn = reparam_conv_bn(nn, channel, kernel_size=3, strides=2, use_bias=use_bias, use_identity=False, activation=activation, name=cur_name)
            else:
                nn = conv_bn(nn, channel, kernel_size=3, strides=2, use_bias=use_bias, activation=activation, name=cur_name)
        csp_expansion = csp_expansions[stack_id] if isinstance(csp_expansions, (list, tuple)) else csp_expansions
        parallel_mode = csp_parallel_mode[stack_id] if isinstance(csp_parallel_mode, (list, tuple)) else csp_parallel_mode
        nn = csp_with_2_conv(nn, depth=depth, expansion=csp_expansion, parallel_mode=parallel_mode, **block_kwargs, name=stack_name + "c2f_")

        if not is_classification_model and stack_id == len(depthes) - 1:
            nn = spatial_pyramid_pooling_fast(nn, pool_size=5, activation=activation, name=stack_name + "spp_fast_")
        features.append(nn)

    if is_classification_model:
        nn = conv_bn(nn, 1280, kernel_size=1, strides=1, activation=activation, name="pre_")
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        outputs = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    else:
        outputs = [features[ii] for ii in out_features]
    model = models.Model(inputs, outputs, name=model_name)

    if is_classification_model:
        add_pre_post_process(model, rescale_mode="raw01")
        reload_model_weights(model, PRETRAINED_DICT, "yolov8", pretrained)
    return model


""" path aggregation fpn """


def path_aggregation_fpn(features, depth=3, parallel_mode=True, use_reparam_conv=False, activation="swish", name=""):
    # yolov8
    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    upsamples = [features[-1]]
    p_name = "p{}_".format(len(features) + 2)
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, feature in enumerate(features[:-1][::-1]):
        cur_p_name = "p{}".format(len(features) + 1 - id)
        p_name = cur_p_name + p_name
        size = functional.shape(feature)[1:-1] if image_data_format() == "channels_last" else functional.shape(feature)[2:]
        nn = functional.resize(upsamples[-1], size, method="nearest")
        nn = functional.concat([nn, feature], axis=channel_axis)

        out_channel = feature.shape[channel_axis]
        nn = csp_with_2_conv(
            nn, out_channel, depth, shortcut=False, parallel_mode=parallel_mode, use_reparam_conv=use_reparam_conv, activation=activation, name=name + p_name
        )
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        nn = conv_bn(downsamples[-1], downsamples[-1].shape[channel_axis], kernel_size=3, strides=2, activation=activation, name=cur_name + "down_")
        nn = functional.concat([nn, ii], axis=channel_axis)

        out_channel = ii.shape[channel_axis]
        nn = csp_with_2_conv(
            nn, out_channel, depth=depth, shortcut=False, parallel_mode=parallel_mode, use_reparam_conv=False, activation=activation, name=cur_name
        )
        downsamples.append(nn)
    return downsamples


""" headers """


def yolov8_head(
    inputs,
    num_classes=80,
    regression_len=64,
    num_anchors=1,
    depth=2,
    hidden_channels=-1,
    use_object_scores=False,
    activation="swish",
    classifier_activation="sigmoid",
    name="",
):
    channel_axis = -1 if image_data_format() == "channels_last" else 1

    outputs = []
    if hidden_channels == -1:
        reg_channel = max(64, regression_len, inputs[0].shape[channel_axis] // 4)
        cls_channel = max(num_classes, inputs[0].shape[channel_axis])
        reg_channels, cls_channels = [reg_channel] * len(inputs), [cls_channel] * len(inputs)
    elif isinstance(hidden_channels, (list, tuple)):
        reg_channels, cls_channels = hidden_channels, hidden_channels
    else:
        reg_channels = cls_channels = hidden_channels

    for id, (feature, reg_channel, cls_channel) in enumerate(zip(inputs, reg_channels, cls_channels)):
        cur_name = name + "{}_".format(id + 1)

        reg_nn = feature
        for id in range(depth):
            reg_nn = conv_bn(reg_nn, reg_channel, 3, activation=activation, name=cur_name + "reg_{}_".format(id + 1))
        reg_out = conv2d_no_bias(reg_nn, regression_len * num_anchors, 1, use_bias=True, bias_initializer="ones", name=cur_name + "reg_{}_".format(depth + 1))

        strides = 2 ** (id + 3)
        bias_init = initializers.constant(math.log(5 / num_classes / (640 / strides) ** 2))
        cls_nn = feature
        for id in range(depth):
            cls_nn = conv_bn(cls_nn, cls_channel, 3, activation=activation, name=cur_name + "cls_{}_".format(id + 1))
        cls_out = conv2d_no_bias(cls_nn, num_classes * num_anchors, 1, use_bias=True, bias_initializer=bias_init, name=cur_name + "cls_{}_".format(depth + 1))
        if classifier_activation is not None:
            cls_out = activation_by_name(cls_out, classifier_activation, name=cur_name + "classifier_")

        # obj_preds, not using for yolov8
        if use_object_scores:
            bias_init = initializers.constant(-math.log((1 - 0.01) / 0.01))
            obj_out = conv2d_no_bias(reg_nn, 1 * num_anchors, kernel_size=1, use_bias=True, bias_initializer=bias_init, name=cur_name + "object_")
            obj_out = activation_by_name(obj_out, classifier_activation, name=cur_name + "object_out_")
            out = functional.concat([reg_out, cls_out, obj_out], axis=channel_axis)
        else:
            out = functional.concat([reg_out, cls_out], axis=channel_axis)
        out = out if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(out)
        out = layers.Reshape([-1, out.shape[-1]], name=cur_name + "output_reshape")(out)
        outputs.append(out)
    outputs = functional.concat(outputs, axis=1)
    return outputs


def yolov8_seg_head(inputs, depth=2, hidden_channels=-1, num_masks=32, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    hidden_channels = hidden_channels if hidden_channels > 0 else max(64, inputs[0].shape[channel_axis])

    """ mask_protos """
    mask_protos = inputs[0]
    mask_protos = conv_bn(mask_protos, hidden_channels, 3, activation=activation, name=name + "mask_protos_1_")
    mask_protos = layers.Conv2DTranspose(hidden_channels, kernel_size=2, strides=2, padding="VALID", name=name + "mask_protos_up")(mask_protos)
    mask_protos = conv_bn(mask_protos, hidden_channels, 3, activation=activation, name=name + "mask_protos_2_")
    mask_protos = conv_bn(mask_protos, num_masks, 1, activation=activation, name=name + "mask_protos_3_")
    mask_protos = mask_protos if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(mask_protos)

    """ mask_coefficients """
    mask_coefficients = []
    mask_hidden_channels = max(inputs[0].shape[channel_axis] // 4, num_masks)
    for id, feature in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        for id in range(depth):
            feature = conv_bn(feature, mask_hidden_channels, 3, activation=activation, name=cur_name + "mask_coefficients_{}_".format(id + 1))
        feature = conv2d_no_bias(feature, num_masks, use_bias=True, bias_initializer="ones", name=cur_name + "mask_coefficients_{}_".format(depth + 1))
        feature = feature if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(feature)
        feature = layers.Reshape([-1, feature.shape[-1]], name=cur_name + "mask_coefficients_reshape")(feature)
        mask_coefficients.append(feature)
    return functional.concat(mask_coefficients, axis=1), mask_protos


""" YOLOV8 models """


def YOLOV8(
    backbone=None,
    csp_channels=[32, 64, 128, 256],  # [YOLOV8Backbone parameters]
    csp_depthes=[1, 2, 2, 1],
    features_pick=[-3, -2, -1],  # [Detector parameters]
    regression_len=64,  # bbox output len, typical value is 4, for yolov8 reg_max=16 -> regression_len = 16 * 4 == 64
    use_reparam_conv=False,  # Use reparam_conv_bn instead of conv_bn block in all csp_blocks.
    paf_parallel_mode=True,  # paf_parallel_mode=False for YOLOV8_X6 `path_aggregation_fpn` module, only concat `short` and the last `deep` one.
    anchors_mode="yolov8",
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" / "yolov8" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    segment_num_masks=0,  # [Segmentation parameters] Set > 0 value like 32 for building model using segment header
    input_shape=(640, 640, 3),  # [Common parameters]
    num_classes=80,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolov8",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
    kwargs=None,  # Not using, recieving parameter
):
    if backbone is None:
        backbone = YOLOV8Backbone(
            csp_channels, csp_depthes, features_pick, use_reparam_conv=use_reparam_conv, input_shape=input_shape, activation=activation, model_name="backbone"
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

    fpn_features = path_aggregation_fpn(features, csp_depthes[-1], paf_parallel_mode, use_reparam_conv=use_reparam_conv, activation=activation, name="pafpn_")

    header_kwargs = {"use_object_scores": use_object_scores, "activation": activation, "classifier_activation": classifier_activation}
    if segment_num_masks > 0:
        mask_coefficients, mask_protos = yolov8_seg_head(fpn_features, num_masks=segment_num_masks, activation=activation, name="seg_head_")
        detect_out = yolov8_head(fpn_features, num_classes, regression_len, num_anchors, **header_kwargs, name="seg_head_")
        detect_mask_out = functional.concat([detect_out, mask_coefficients], axis=-1)  # detect_out, mask_coefficients needs to apply NMS together
        outputs = [
            layers.Activation("linear", dtype="float32", name="detect_mask_outputs_fp32")(detect_mask_out),
            layers.Activation("linear", dtype="float32", name="mask_protos_outputs_fp32")(mask_protos),
        ]
    else:
        outputs = yolov8_head(fpn_features, num_classes, regression_len, num_anchors, **header_kwargs, name="head_")
        outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)

    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolov8", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = eval_func.DecodePredictions(
        backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale, regression_len=regression_len, num_masks=segment_num_masks
    )
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    return model


def switch_to_deploy(model):
    from keras_cv_attention_models.model_surgery.model_surgery import fuse_reparam_blocks, convert_to_fused_conv_bn_model

    new_model = convert_to_fused_conv_bn_model(fuse_reparam_blocks(convert_to_fused_conv_bn_model(model)))

    # Has to create a new one, or will `ValueError: Unable to create group (name already exists)` when saving
    post_process = eval_func.DecodePredictions(
        input_shape=model.decode_predictions.__input_shape__,
        pyramid_levels=model.decode_predictions.pyramid_levels,
        anchors_mode=model.decode_predictions.anchors_mode,
        use_object_scores=model.decode_predictions.use_object_scores,
        anchor_scale=model.decode_predictions.anchor_scale,
        regression_len=model.decode_predictions.regression_len,
    )
    add_pre_post_process(new_model, rescale_mode=model.preprocess_input.rescale_mode, post_process=post_process)
    return new_model


""" Detection models """


@register_model
def YOLOV8_N(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_n"), **kwargs)


@register_model
def YOLOV8_S(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [64, 128, 256, 512]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_s"), **kwargs)


@register_model
def YOLOV8_M(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [96, 192, 384, 576]
    csp_depthes = [2, 4, 4, 2]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_m"), **kwargs)


@register_model
def YOLOV8_L(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [128, 256, 512, 512]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_l"), **kwargs)


@register_model
def YOLOV8_X(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [160, 320, 640, 640]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_x"), **kwargs)


@register_model
def YOLOV8_X6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [160, 320, 640, 640, 640]
    csp_depthes = [3, 6, 6, 3, 3]
    features_pick = [-4, -3, -2, -1]
    paf_parallel_mode = False  # C2
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_x6"), **kwargs)


""" Classification models """


@register_model
def YOLOV8_N_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_n_cls"), **kwargs)


@register_model
def YOLOV8_S_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [64, 128, 256, 512]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_s_cls"), **kwargs)


@register_model
def YOLOV8_M_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [96, 192, 384, 768]
    depthes = [2, 4, 4, 2]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_m_cls"), **kwargs)


@register_model
def YOLOV8_L_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [128, 256, 512, 1024]
    depthes = [3, 6, 6, 3]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_l_cls"), **kwargs)


@register_model
def YOLOV8_X_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [160, 320, 640, 1280]
    depthes = [3, 6, 6, 3]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_x_cls"), **kwargs)


""" Segmentation models """


@register_model
def YOLOV8_N_SEG(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOV8(**locals(), segment_num_masks=32, model_name=kwargs.pop("model_name", "yolov8_n_seg"), **kwargs)


@register_model
def YOLOV8_S_SEG(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [64, 128, 256, 512]
    return YOLOV8(**locals(), segment_num_masks=32, model_name=kwargs.pop("model_name", "yolov8_s_seg"), **kwargs)


@register_model
def YOLOV8_M_SEG(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [96, 192, 384, 576]
    csp_depthes = [2, 4, 4, 2]
    return YOLOV8(**locals(), segment_num_masks=32, model_name=kwargs.pop("model_name", "yolov8_m_seg"), **kwargs)


@register_model
def YOLOV8_L_SEG(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [128, 256, 512, 512]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), segment_num_masks=32, model_name=kwargs.pop("model_name", "yolov8_l_seg"), **kwargs)


@register_model
def YOLOV8_X_SEG(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [160, 320, 640, 640]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), segment_num_masks=32, model_name=kwargs.pop("model_name", "yolov8_x_seg"), **kwargs)
