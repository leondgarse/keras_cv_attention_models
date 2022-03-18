import tensorflow as tf
from tensorflow import keras
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
from keras_cv_attention_models.coco.eval_func import DecodePredictions
from keras_cv_attention_models.coco import anchors_func

PRETRAINED_DICT = {"yolor_csp": {"coco": "ed0aa82a07c4e65e9cd3d2e6ad2d0548"}, "yolor_csp_x": {"coco": "615125ce1cd1c855f8045bf079456598"}}


""" CSPDarknet backbone """
BATCH_NORM_EPSILON = 1e-4
BATCH_NORM_MOMENTUM = 0.03


def conv_dw_pw_block(inputs, filters, kernel_size=1, strides=1, use_depthwise_conv=False, activation="swish", name=""):
    nn = inputs
    if use_depthwise_conv:
        nn = depthwise_conv2d_no_bias(nn, kernel_size, strides, padding="SAME", name=name)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "dw_")
        kernel_size, strides = 1, 1
    nn = conv2d_no_bias(nn, filters, kernel_size, strides, padding="SAME", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)
    return nn


def csp_block(inputs, expansion=0.5, use_shortcut=True, use_depthwise_conv=False, activation="swish", name=""):
    input_channels = inputs.shape[-1]
    nn = conv_dw_pw_block(inputs, int(input_channels * expansion), activation=activation, name=name + "1_")
    nn = conv_dw_pw_block(nn, input_channels, kernel_size=3, strides=1, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "2_")
    if use_shortcut:
        nn = keras.layers.Add()([inputs, nn])
    return nn


def csp_stack(inputs, depth, out_channels=-1, expansion=0.5, use_shortcut=True, use_pre_mode=False, use_depthwise_conv=False, activation="swish", name=""):
    out_channels = inputs.shape[-1] if out_channels == -1 else out_channels
    hidden_channels = int(out_channels * expansion)
    if use_pre_mode:
        inputs = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_")
    short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = inputs if use_pre_mode else conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_pre_")
    for id in range(depth):
        block_name = name + "block{}_".format(id + 1)
        deep = csp_block(deep, 1, use_shortcut=use_shortcut, use_depthwise_conv=use_depthwise_conv, activation=activation, name=block_name)
    if not use_pre_mode:
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_post_")

    out = tf.concat([deep, short], axis=-1)
    out = conv_dw_pw_block(out, out_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def res_spatial_pyramid_pooling(inputs, depth, expansion=0.5, pool_sizes=(5, 9, 13), activation="swish", name=""):
    input_channels = inputs.shape[-1]
    hidden_channels = int(input_channels * expansion)
    short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_1_")
    deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_2_")
    deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_3_")
    pp = [keras.layers.MaxPooling2D(pool_size=ii, strides=1, padding="SAME")(deep) for ii in pool_sizes]
    deep = tf.concat([deep, *pp][::-1], axis=-1)  # yolor_csp.cfg, SSP concat layers=-1,-3,-5,-6
    for id in range(depth - 1):  # First one is `pre`
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "post_{}_".format(id * 2 + 1))
        deep = conv_dw_pw_block(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "post_{}_".format(id * 2 + 2))

    out = tf.concat([deep, short], axis=-1)
    out = conv_dw_pw_block(out, hidden_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def CSPDarknet(width_mul=1, depth_mul=1, out_features=[-3, -2, -1], use_depthwise_conv=False, input_shape=(512, 512, 3), activation="swish", model_name=""):
    base_channels = int(width_mul * 64)
    inputs = keras.layers.Input(input_shape)

    """ Stem """
    nn = conv_dw_pw_block(inputs, 32, kernel_size=3, strides=1, activation=activation, name="stem_1_")  # Fixed as 32
    nn = conv_dw_pw_block(nn, base_channels, kernel_size=3, strides=2, activation=activation, name="stem_2_")
    nn = csp_block(nn, expansion=0.5, activation=activation, name="stem_3_")
    features = [nn]

    """ dark blocks """
    depthes = [max(round(depth_mul * ii), 1) for ii in [2, 8, 8, 4]]  # YOLOR_CSP depth
    channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
    use_spps = [False, False, False, True]
    for id, (channel, depth, use_spp) in enumerate(zip(channels, depthes, use_spps)):
        stack_name = "stack{}_".format(id + 1)
        nn = conv_dw_pw_block(nn, channel, 3, strides=2, use_depthwise_conv=use_depthwise_conv, activation=activation, name=stack_name + "downsample_")
        nn = csp_stack(nn, depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=stack_name)
        if use_spp:
            ssp_depth = max(round(depth_mul * 2), 1)
            nn = res_spatial_pyramid_pooling(nn, ssp_depth, activation=activation, name=stack_name + "spp_")
        features.append(nn)

    nn = [features[ii] for ii in out_features]
    model = keras.models.Model(inputs, nn, name=model_name)
    return model


""" path aggregation fpn """


def upsample_merge(inputs, csp_depth, use_depthwise_conv=False, activation="swish", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    upsample = conv_dw_pw_block(inputs[-1], inputs[0].shape[-1], activation=activation, name=name + "up_")

    # inputs[0] = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=name + "up")(fpn_out)
    inputs[-1] = tf.image.resize(upsample, tf.shape(inputs[0])[1:-1], method="nearest")
    nn = tf.concat(inputs, axis=-1)
    nn = csp_stack(nn, csp_depth, nn.shape[-1] // 2, 1.0, False, use_pre_mode=True, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name)
    return nn


def downsample_merge(inputs, csp_depth, use_depthwise_conv=False, activation="swish", name=""):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    inputs[0] = conv_dw_pw_block(inputs[0], inputs[-1].shape[-1], 3, 2, use_depthwise_conv, activation=activation, name=name + "down_")
    nn = tf.concat(inputs, axis=-1)
    nn = csp_stack(nn, csp_depth, nn.shape[-1] // 2, 1.0, False, use_pre_mode=True, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name)
    return nn


def path_aggregation_fpn(features, depth_mul=1, use_depthwise_conv=False, activation="swish", name=""):
    # p5 ─────┬───> pan_out0
    #         ↓         ↑
    # p4 ─> p4p5 ─> pan_out1
    #         ↓         ↑
    # p3 ─> pan_out2 ───┘
    csp_depth = max(round(depth_mul * 2), 1)
    p3, p4, p5 = features  # p3: [64, 64, 256], p4: [32, 32, 512], p5: [16, 16, 512]
    p3 = conv_dw_pw_block(p3, p3.shape[-1] // 2, kernel_size=1, activation=activation, name=name + "p3_down_")  # [64, 64, 128]
    p4 = conv_dw_pw_block(p4, p4.shape[-1] // 2, kernel_size=1, activation=activation, name=name + "p4_down_")  # [32, 32, 256]

    # p4p5: [32, 32, 256]
    p4p5 = upsample_merge([p4, p5], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "p4p5_")
    # pan_out2: [64, 64, 128]
    pan_out2 = upsample_merge([p3, p4p5], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "p3p4p5_")
    # pan_out1: [32, 32, 256]
    pan_out1 = downsample_merge([pan_out2, p4p5], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3n3_")
    # pan_out0: [16, 16, 512]
    pan_out0 = downsample_merge([pan_out1, p5], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3n4_")
    return [pan_out2, pan_out1, pan_out0]


""" YOLORHead """


def yolor_head_single(inputs, filters, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", name=""):
    initializer = tf.initializers.truncated_normal(stddev=0.2)

    nn = conv_dw_pw_block(inputs, filters, 3, activation=activation, name=name + "1_")
    nn = BiasLayer(initializer=initializer, name=name + "shift_channel")(nn)

    ouput_classes = num_classes + (5 if use_object_scores else 4)  # num_anchors = 3, num_anchors * (80 + 5) = 255
    nn = keras.layers.Conv2D(ouput_classes * num_anchors, kernel_size=1, name=name + "2_conv")(nn)
    control_channels_layer = ChannelAffine(use_bias=False, name=name + "control_channel")
    control_channels_layer.ww_init = initializer
    nn = control_channels_layer(nn)
    # return nn
    return keras.layers.Reshape([-1, ouput_classes], name=name + "output_reshape")(nn)


def yolor_head(inputs, num_classes=80, num_anchors=1, use_object_scores=True, activation="swish", classifier_activation="sigmoid", name=""):
    outputs = []
    for id, input in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        # filters = int(input.shape[-1] * 2 * width_mul)
        filters = int(input.shape[-1] * 2)
        out = yolor_head_single(input, filters, num_classes, num_anchors, use_object_scores, activation=activation, name=cur_name)
        outputs.append(out)
    # return outputs
    outputs = tf.concat(outputs, axis=1)
    return activation_by_name(outputs, classifier_activation, name="classifier_")


""" YOLOR models """


def YOLOR(
    backbone=None,
    features_pick=[-3, -2, -1],
    depth_mul=1,
    width_mul=1,  # width_mul is for CSPDarknet backbone only, not for yolor_head
    use_depthwise_conv=False,
    use_anchor_free_mode=False,
    num_anchors="auto",  # "auto" means 1 if use_anchor_free_mode else 3
    use_object_scores=True,  # "auto" means same with use_anchor_free_mode
    input_shape=(640, 640, 3),
    num_classes=80,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolor",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if use_anchor_free_mode else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
    kwargs=None,  # Not using, recieving parameter
):
    if backbone is None:
        width_mul = width_mul if width_mul > 0 else 1
        backbone = CSPDarknet(width_mul, depth_mul, features_pick, use_depthwise_conv, input_shape, activation=activation, model_name="darknet")
        features = backbone.outputs
    else:
        if isinstance(features_pick[0], str):
            features = [backbone.get_layer(layer_name) for layer_name in features_pick]
        else:
            features = model_surgery.get_pyramide_feture_layers(backbone)
            features = [features[id] for id in features_pick]
        print(">>>> features:", {ii.name: ii.output_shape for ii in features})
        features = [ii.output for ii in features]
        # width_mul = width_mul if width_mul > 0 else min([ii.shape[-1] for ii in features]) / 256
        # print(">>>> width_mul:", width_mul)

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True

    inputs = backbone.inputs[0]
    use_object_scores = use_anchor_free_mode if use_object_scores == "auto" else use_object_scores
    num_anchors = (1 if use_anchor_free_mode else 3) if num_anchors == "auto" else num_anchors
    fpn_features = path_aggregation_fpn(features, depth_mul=depth_mul, use_depthwise_conv=use_depthwise_conv, activation=activation, name="pafpn_")
    outputs = yolor_head(fpn_features, num_classes, num_anchors, use_object_scores, activation, classifier_activation, name="head_")
    outputs = keras.layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = keras.models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolor", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = YolorDecodePredictions(backbone.input_shape[1:], pyramid_levels, use_object_scores, use_anchor_free_mode)
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


class YolorDecodePredictions(DecodePredictions):
    def __init__(self, input_shape=640, pyramid_levels=[3, 5], use_object_scores=True, use_anchor_free_mode=False):
        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        self.use_object_scores, self.use_anchor_free_mode = use_object_scores, use_anchor_free_mode
        if input_shape is not None and (isinstance(input_shape, (list, tuple)) and input_shape[0] is not None):
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None

    def __init_anchor__(self, input_shape):
        input_shape = input_shape[:2] if isinstance(input_shape, (list, tuple)) else (input_shape, input_shape)
        if self.use_anchor_free_mode:
            self.anchors = anchors_func.get_anchor_free_anchors(input_shape, self.pyramid_levels)
        else:
            self.anchors = anchors_func.get_yolor_anchors(input_shape, self.pyramid_levels)


def YOLOR_CSP(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOR(**locals(), depth_mul=1.0, width_mul=1.0, model_name=kwargs.pop("model_name", "yolor_csp"), **kwargs)


def YOLOR_CSPX(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOR(**locals(), depth_mul=1.3, width_mul=1.25, model_name=kwargs.pop("model_name", "yolor_csp_x"), **kwargs)
