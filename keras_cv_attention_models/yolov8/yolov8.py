import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
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
    "yolov8_l": {"coco": "db0fcde5d2811b33b7f5f0f400d76911"},
    "yolov8_m": {"coco": "cb8c25148bb17485776ade4cf80cc6f6"},
    "yolov8_n": {"coco": "4cb83c7e452cdcd440b75546df0b211e"},
    "yolov8_s": {"coco": "4e1ac133e2a8831845172d8491c2747a"},
    "yolov8_x": {"coco": "2be28e650bf299aeea7ee26ab765a23e"},
    "yolov8_l_cls": {"imagenet": "071f41125034dd15401f6c6925fc1e6f"},
    "yolov8_m_cls": {"imagenet": "35ef50aa07ff232afa08f321447e354d"},
    "yolov8_n_cls": {"imagenet": "b1cfac787589689c0f2abde6893ec140"},
    "yolov8_s_cls": {"imagenet": "2caa57e8cf67b39921c35f89cea5061c"},
    "yolov8_x_cls": {"imagenet": "2d4b8b996c24f5fde903678ee8b7cf20"},
}


""" Yolov8Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(inputs, output_channel, kernel_size=1, strides=1, activation="swish", name=""):
    # print(f">>>> {inputs.shape = }, {output_channel = }, {kernel_size = }, {strides = }")
    nn = conv2d_no_bias(inputs, output_channel, kernel_size, strides, padding="SAME", name=name)
    return batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)


def csp_with_2_conv(inputs, channels=-1, depth=2, shortcut=True, expansion=0.5, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    pre = conv_bn(inputs, hidden_channels * 2, kernel_size=1, activation=activation, name=name + "pre_")
    short, deep = functional.split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_{}_1_".format(id))
        deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_{}_2_".format(id))
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep)
    out = functional.concat(out, axis=channel_axis)
    out = conv_bn(out, channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def spatial_pyramid_pooling_fast(inputs, pool_size=5, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    nn = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_")
    pool_1 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(nn)
    pool_2 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(pool_1)
    pool_3 = layers.MaxPool2D(pool_size=pool_size, strides=1, padding="SAME")(pool_2)

    out = functional.concat([nn, pool_1, pool_2, pool_3], axis=channel_axis)
    out = conv_bn(out, input_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def YOLOV8Backbone(
    channels=[32, 64, 128, 256],
    depthes=[1, 2, 2, 1],
    out_features=[-3, -2, -1],
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
    channel_axis = -1 if image_data_format() == "channels_last" else 1
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
    nn = conv_bn(inputs, stem_width // 2, kernel_size=3, strides=2, activation=activation, name="stem_1_")
    nn = conv_bn(nn, stem_width, kernel_size=3, strides=2, activation=activation, name="stem_2_")

    """ blocks """
    features = [nn]
    for stack_id, (channel, depth) in enumerate(zip(channels, depthes)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id >= 1:
            nn = conv_bn(nn, channel, kernel_size=3, strides=2, activation=activation, name=stack_name + "downsample_")
        nn = csp_with_2_conv(nn, depth=depth, expansion=0.5, activation=activation, name=stack_name + "c2f_")

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


def path_aggregation_fpn(features, depth=3, activation="swish", name=""):
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
        nn = csp_with_2_conv(nn, channels=out_channel, depth=depth, shortcut=False, activation=activation, name=name + p_name)
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        nn = conv_bn(downsamples[-1], downsamples[-1].shape[channel_axis], kernel_size=3, strides=2, activation=activation, name=cur_name + "down_")
        nn = functional.concat([nn, ii], axis=channel_axis)

        out_channel = ii.shape[channel_axis]
        nn = csp_with_2_conv(nn, channels=out_channel, depth=depth, shortcut=False, activation=activation, name=cur_name)
        downsamples.append(nn)
    return downsamples


def yolov8_head(
    inputs, num_classes=80, regression_len=64, num_anchors=1, use_object_scores=False, activation="swish", classifier_activation="sigmoid", name=""
):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    bias_init = initializers.constant(-math.log((1 - 0.01) / 0.01))

    outputs = []
    reg_channels = max(16, regression_len, inputs[0].shape[channel_axis] // 4)
    cls_channels = max(num_classes, inputs[0].shape[channel_axis])
    for id, feature in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)

        reg_nn = conv_bn(feature, reg_channels, 3, activation=activation, name=cur_name + "reg_1_")
        reg_nn = conv_bn(reg_nn, reg_channels, 3, activation=activation, name=cur_name + "reg_2_")
        reg_out = conv2d_no_bias(reg_nn, regression_len * num_anchors, 1, use_bias=True, name=cur_name + "reg_3_")

        cls_nn = conv_bn(feature, cls_channels, 3, activation=activation, name=cur_name + "cls_1_")
        cls_nn = conv_bn(cls_nn, cls_channels, 3, activation=activation, name=cur_name + "cls_2_")
        cls_out = conv2d_no_bias(cls_nn, num_classes * num_anchors, 1, use_bias=True, name=cur_name + "cls_3_")
        if classifier_activation is not None:
            cls_out = activation_by_name(cls_out, classifier_activation, name=cur_name + "classifier_")

        # obj_preds
        if use_object_scores:
            obj_out = conv2d_no_bias(reg_nn, 1 * num_anchors, kernel_size=1, use_bias=True, bias_initializer=bias_init, name=cur_name + "object_")
            obj_out = activation_by_name(obj_out, "sigmoid", name=cur_name + "object_out_")
            out = functional.concat([reg_out, cls_out, obj_out], axis=channel_axis)
        else:
            out = functional.concat([reg_out, cls_out], axis=channel_axis)
        out = out if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(out)
        out = layers.Reshape([-1, out.shape[-1]], name=cur_name + "output_reshape")(out)
        outputs.append(out)
    outputs = functional.concat(outputs, axis=1)
    return outputs


""" YOLOV8 models """


def YOLOV8(
    backbone=None,
    csp_channels=[32, 64, 128, 256],  # [YOLOV8Backbone parameters]
    csp_depthes=[1, 2, 2, 1],
    features_pick=[-3, -2, -1],  # [Detector parameters]
    regression_len=64,  # bbox output len, typical value is 4, for yolov8 reg_max=16 -> regression_len = 16 * 4 == 64
    anchors_mode="yolov8",
    num_anchors="auto",  # "auto" means: anchors_mode=="anchor_free" / "yolov8" -> 1, anchors_mode=="yolor" -> 3, else 9
    use_object_scores="auto",  # "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False
    input_shape=(640, 640, 3),
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
    # Regard input_shape as force using original shape if first element is None or -1,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)

    if backbone is None:
        backbone = YOLOV8Backbone(
            channels=csp_channels, depthes=csp_depthes, out_features=features_pick, input_shape=input_shape, activation=activation, model_name="backbone"
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

    fpn_features = path_aggregation_fpn(features, depth=csp_depthes[-1], activation=activation, name="pafpn_")

    outputs = yolov8_head(fpn_features, num_classes, regression_len, num_anchors, use_object_scores, activation, classifier_activation, name="head_")
    outputs = layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolov8", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = eval_func.DecodePredictions(
        backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale, regression_len=regression_len
    )
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


def YOLOV8_N(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_n"), **kwargs)


def YOLOV8_S(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [64, 128, 256, 512]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_s"), **kwargs)


def YOLOV8_M(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [96, 192, 384, 576]
    csp_depthes = [2, 4, 4, 2]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_m"), **kwargs)


def YOLOV8_L(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [128, 256, 512, 512]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_l"), **kwargs)


def YOLOV8_X(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [160, 320, 640, 640]
    csp_depthes = [3, 6, 6, 3]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_x"), **kwargs)


def YOLOV8_X6(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    csp_channels = [160, 320, 640, 640, 640]
    csp_depthes = [3, 6, 6, 3, 3]
    features_pick = [-4, -3, -2, -1]
    return YOLOV8(**locals(), model_name=kwargs.pop("model_name", "yolov8_x6"), **kwargs)


""" Classification models """


def YOLOV8_N_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_n_cls"), **kwargs)


def YOLOV8_S_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [64, 128, 256, 512]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_s_cls"), **kwargs)


def YOLOV8_M_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [96, 192, 384, 768]
    depthes = [2, 4, 4, 2]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_m_cls"), **kwargs)


def YOLOV8_L_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [128, 256, 512, 1024]
    depthes = [3, 6, 6, 3]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_l_cls"), **kwargs)


def YOLOV8_X_CLS(input_shape=(640, 640, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    channels = [160, 320, 640, 1280]
    depthes = [3, 6, 6, 3]
    return YOLOV8Backbone(**locals(), model_name=kwargs.pop("model_name", "yolov8_x_cls"), **kwargs)
