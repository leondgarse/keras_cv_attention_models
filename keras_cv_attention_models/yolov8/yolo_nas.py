from keras_cv_attention_models.backend import layers, functional, models, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import add_pre_post_process
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco import eval_func, anchors_func
from keras_cv_attention_models.yolov8.yolov8 import YOLOV8Backbone, conv_bn, csp_with_2_conv, yolov8_head, switch_to_deploy

PRETRAINED_DICT = {
    "yolo_nas_l": {"coco": "cea25dd86e8e4aa1fd82b7e1288fa583"},
    "yolo_nas_m": {"coco": "16cbc4683b51894334b0264def1593a2"},
    "yolo_nas_s": {"coco": "283395afacb7ca5ea597d2e48dd19329"},
    "yolo_nas_l_before_reparam": {"coco": "a22ce5b0a036cf27a9e32fc80f09c373"},
    "yolo_nas_m_before_reparam": {"coco": "d093634fc5af38da0e763512ac829da4"},
    "yolo_nas_s_before_reparam": {"coco": "8c41ba972fc1bc66810c8f5ee15e404e"},
}


""" path aggregation fpn """


def upsample_merge(inputs, csp_depth=2, expansion=0.5, parallel_mode=False, use_alpha=True, use_reparam_conv=False, activation="relu", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    low, middle, high = inputs
    channels = high.shape[channel_axis]

    fpn_out = conv_bn(low, channels, activation=activation, name=name + "fpn_")
    low = layers.Conv2DTranspose(channels, kernel_size=2, strides=2, padding="valid", name=name + "up_conv_transpose")(fpn_out)

    middle = conv_bn(middle, channels, activation=activation, name=name + "middle_")
    high = conv_bn(high, channels, activation=activation, name=name + "high_")
    high = conv_bn(high, channels, kernel_size=3, strides=2, activation=activation, name=name + "high_down_")

    nn = functional.concat([low, middle, high], axis=channel_axis)
    nn = conv_bn(nn, channels, kernel_size=1, strides=1, activation=activation, name=name + "reduce_")
    shortcut, use_bias = True, True
    nn = csp_with_2_conv(nn, channels, csp_depth, shortcut, expansion, parallel_mode, use_bias, use_alpha, use_reparam_conv, activation=activation, name=name)
    return fpn_out, nn


def downsample_merge(inputs, csp_depth=2, expansion=0.5, parallel_mode=False, use_alpha=True, use_reparam_conv=False, activation="relu", name=""):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    # target_channel = inputs[-1].shape[channel_axis]
    inputs[0] = conv_bn(inputs[0], inputs[-1].shape[channel_axis], kernel_size=3, strides=2, activation=activation, name=name + "down_")
    nn = functional.concat(inputs, axis=channel_axis)
    channels = nn.shape[channel_axis]
    shortcut, use_bias = True, False
    nn = csp_with_2_conv(nn, channels, csp_depth, shortcut, expansion, parallel_mode, use_bias, use_alpha, use_reparam_conv, activation=activation, name=name)
    return nn


def path_aggregation_fpn(
    features, output_channels=[64, 128, 256], depthes=2, expansions=0.5, parallel_mode=False, use_alpha=True, use_reparam_conv=False, activation="relu", name=""
):
    # p5 ---+-> fpn_out0 -----------------------+-> pan_out0
    #       v [up -> concat]                    ^ [down -> concat]
    # p4 ---+-> f_out0 --------+-> fpn_out1 ----+-> pan_out1
    #       ^ [down -> concat] v [up -> concat] ^ [down -> concat]
    # p3 ---+------------------+----------------+-> pan_out2
    #                          ^ [down -> concat]
    # p2 ----------------------+
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    expansions = expansions if isinstance(expansions, (list, tuple)) else [expansions] * 4
    depthes = depthes if isinstance(depthes, (list, tuple)) else [depthes] * 4

    p2, p3, p4, p5 = features  # p2: [160, 160, 96], p3: [80, 80, 192], p4: [40, 40, 384], p5: [20, 20, 768]
    fpn_out0, f_out0 = upsample_merge([p5, p4, p3], depthes[0], expansions[0], parallel_mode, use_alpha, use_reparam_conv, activation, name=name + "p3p4p5_")
    fpn_out1, pan_out2 = upsample_merge(
        [f_out0, p3, p2], depthes[1], expansions[1], parallel_mode, use_alpha, use_reparam_conv, activation, name=name + "c3p3c2_"
    )

    use_reparam_conv = False  # False for downsample_merge blocks
    pan_out1 = downsample_merge([pan_out2, fpn_out1], depthes[2], expansions[2], parallel_mode, use_alpha, use_reparam_conv, activation, name=name + "c3n3_")
    pan_out0 = downsample_merge([pan_out1, fpn_out0], depthes[3], expansions[3], parallel_mode, use_alpha, use_reparam_conv, activation, name=name + "c3n4_")

    out = []
    for id, (pan_out, channel) in enumerate(zip([pan_out2, pan_out1, pan_out0], output_channels)):
        nn = conv_bn(pan_out, channel, activation=activation, name=name + "out{}_".format(id + 1))
        out.append(nn)
    return out


""" YOLO_NAS models """


def YOLO_NAS(
    backbone=None,
    csp_channels=[96, 192, 384, 768],  # [YOLOV8Backbone parameters]
    csp_depthes=[2, 3, 5, 2],
    features_pick=[-3, -2, -1],  # [Detector parameters]
    csp_expansions=[1 / 3, 1 / 3, 0.25, 0.25],
    paf_expansions=[1 / 3, 0.5, 1 / 3, 1 / 6],
    regression_len=68,  # bbox output len, typical value is 4, for yolov8 reg_max=16 -> regression_len = 16 * 4 == 64
    use_alpha=True,
    # use_bias=True,  # Actually used for reloading reparametered conv weights
    header_depth=1,
    use_reparam_conv=True,  # Use rep_vgg_conv_bn instead of conv_bn block in all csp_blocks.
    csp_parallel_mode=False,
    paf_parallel_mode=False,
    paf_output_channels=[64, 128, 256],
    paf_depthes=2,
    anchors_mode="yolov8",
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" / "yolov8" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    input_shape=(640, 640, 3),
    num_classes=80,
    activation="relu",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolo_nas",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
    kwargs=None,  # Not using, recieving parameter
):
    if backbone is None:
        features_pick = [-4, -3, -2, -1]
        backbone_kwargs = dict(use_alpha=use_alpha, use_bias=True, use_reparam_conv=use_reparam_conv, input_shape=input_shape, activation=activation)
        backbone = YOLOV8Backbone(csp_channels, csp_depthes, features_pick, csp_expansions, csp_parallel_mode, **backbone_kwargs, model_name="backbone")
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

    fpn_features = path_aggregation_fpn(
        features, paf_output_channels, paf_depthes, paf_expansions, paf_parallel_mode, use_alpha, use_reparam_conv, activation=activation, name="pafpn_"
    )

    header_kwargs = {"depth": header_depth, "use_object_scores": use_object_scores, "activation": activation, "classifier_activation": classifier_activation}
    outputs = yolov8_head(fpn_features, num_classes, regression_len, num_anchors, hidden_channels=paf_output_channels, **header_kwargs, name="head_")
    outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)

    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolov8", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(fpn_features) - 1]  # -> [3, 5], Note: path_aggregation_fpn merged one feature
    post_process = eval_func.DecodePredictions(
        backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale, regression_len=regression_len
    )
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    return model


@register_model
def YOLO_NAS_S(
    input_shape=(640, 640, 3),
    freeze_backbone=False,
    num_classes=80,
    backbone=None,
    use_reparam_conv=True,
    classifier_activation="sigmoid",
    pretrained="coco",
    **kwargs
):
    return YOLO_NAS(**locals(), model_name=kwargs.pop("model_name", "yolo_nas_s_before_reparam" if use_reparam_conv else "yolo_nas_s"), **kwargs)


@register_model
def YOLO_NAS_M(
    input_shape=(640, 640, 3),
    freeze_backbone=False,
    num_classes=80,
    backbone=None,
    use_reparam_conv=True,
    classifier_activation="sigmoid",
    pretrained="coco",
    **kwargs
):
    csp_expansions = [2 / 3, 2 / 3, 2 / 3, 0.5]
    csp_parallel_mode = [True, True, True, False]
    paf_expansions = [1, 2 / 3, 1, 2 / 3]
    paf_output_channels = [96, 192, 384]
    paf_depthes = [2, 3, 2, 3]
    return YOLO_NAS(**locals(), model_name=kwargs.pop("model_name", "yolo_nas_m_before_reparam" if use_reparam_conv else "yolo_nas_m"), **kwargs)


@register_model
def YOLO_NAS_L(
    input_shape=(640, 640, 3),
    freeze_backbone=False,
    num_classes=80,
    backbone=None,
    use_reparam_conv=True,
    classifier_activation="sigmoid",
    pretrained="coco",
    **kwargs
):
    csp_expansions = [1, 2 / 3, 2 / 3, 2 / 3]
    csp_parallel_mode = True
    paf_expansions = [2 / 3, 4 / 3, 2 / 3, 2 / 3]
    paf_output_channels = [128, 256, 512]
    paf_depthes = 4
    return YOLO_NAS(**locals(), model_name=kwargs.pop("model_name", "yolo_nas_l_before_reparam" if use_reparam_conv else "yolo_nas_l"), **kwargs)
