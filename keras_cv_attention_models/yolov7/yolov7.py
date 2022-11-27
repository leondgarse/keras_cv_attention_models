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
from keras_cv_attention_models.coco import eval_func, anchors_func

PRETRAINED_DICT = {"yolov7_csp": {"coco": "52d5def3f37edb0b3c5508d3c5cb8bb0"}}


""" Yolov7Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(inputs, output_channel, kernel_size=1, strides=1, activation="swish", name=""):
    nn = conv2d_no_bias(inputs, output_channel, kernel_size, strides, padding="SAME", name=name)
    return batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)


def stack_6(inputs, filters, concats=[-1, -3, -5, -6], mid_ratio=1.0, out_ratio=4, activation="swish", name=""):
    first = conv_bn(inputs, filters, kernel_size=1, strides=1, activation=activation, name=name + "first_")
    second = conv_bn(inputs, filters, kernel_size=1, strides=1, activation=activation, name=name + "second_")

    mid_filters = int(mid_ratio * filters)
    third = conv_bn(second, mid_filters, kernel_size=3, strides=1, activation=activation, name=name + "third_")
    fourth = conv_bn(third, mid_filters, kernel_size=3, strides=1, activation=activation, name=name + "fourth_")

    fifth = conv_bn(fourth, mid_filters, kernel_size=3, strides=1, activation=activation, name=name + "fifth_")
    sixth = conv_bn(fifth, mid_filters, kernel_size=3, strides=1, activation=activation, name=name + "sixth_")

    gathered = [first, second, third, fourth, fifth, sixth]
    nn = tf.concat([gathered[ii] for ii in concats], axis=-1)  # [sixth, fourth, second, first] by defualt
    nn = conv_bn(nn, int(filters * out_ratio), kernel_size=1, strides=1, activation=activation, name=name + "out_")
    return nn


# Same with yolor
def csp_conv_downsample(inputs, ratio=0.5, activation="swish", name=""):
    input_channel = inputs.shape[-1]
    hidden_channel = int(input_channel * ratio)
    pool_branch = keras.layers.MaxPool2D(pool_size=2, strides=2, name=name + "pool")(inputs)
    pool_branch = conv_bn(pool_branch, hidden_channel, kernel_size=1, strides=1, activation=activation, name=name + "pool_")

    conv_branch = conv_bn(inputs, hidden_channel, kernel_size=1, strides=1, activation=activation, name=name + "conv_1_")
    conv_branch = conv_bn(conv_branch, hidden_channel, kernel_size=3, strides=2, activation=activation, name=name + "conv_2_")

    nn = tf.concat([conv_branch, pool_branch], axis=-1)
    return nn


# Same with yolor
def res_spatial_pyramid_pooling(inputs, depth=2, expansion=0.5, pool_sizes=(5, 9, 13), activation="swish", name=""):
    input_channels = inputs.shape[-1]
    hidden_channels = int(input_channels * expansion)
    short = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = conv_bn(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_1_")
    deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "pre_2_")
    deep = conv_bn(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "pre_3_")
    pp = [keras.layers.MaxPooling2D(pool_size=ii, strides=1, padding="SAME")(deep) for ii in pool_sizes]  # TODO: SAME padding
    deep = tf.concat([deep, *pp], axis=-1)  # yolov7r SPPCSPC concat, different from yolor
    for id in range(depth - 1):  # First one is `pre`
        deep = conv_bn(deep, hidden_channels, kernel_size=1, activation=activation, name=name + "post_{}_".format(id * 2 + 1))
        deep = conv_bn(deep, hidden_channels, kernel_size=3, activation=activation, name=name + "post_{}_".format(id * 2 + 2))

    out = tf.concat([deep, short], axis=-1)
    out = conv_bn(out, hidden_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def Yolov7Backbone(
    channels=[64, 128, 256, 256],
    stem_width=-1,  # -1 means using channels[0]
    out_features=[-3, -2, -1],
    input_shape=(512, 512, 3),
    activation="swish",
    model_name="",
):
    inputs = keras.layers.Input(input_shape)

    """ Stem """
    stem_width = stem_width if stem_width > 0 else channels[0]
    nn = conv_bn(inputs, 32, kernel_size=3, strides=1, activation=activation, name="stem_1_")
    nn = conv_bn(nn, stem_width, kernel_size=3, strides=2, activation=activation, name="stem_2_")
    nn = conv_bn(nn, stem_width, kernel_size=3, strides=1, activation=activation, name="stem_3_")

    """ blocks """
    features = [nn]
    use_csp_conv_downsamples = [False, True, True, True]
    for id, (channel, use_csp_conv_downsample) in enumerate(zip(channels, use_csp_conv_downsamples)):
        stack_name = "stack{}_".format(id + 1)
        if use_csp_conv_downsample:
            nn = csp_conv_downsample(nn, activation=activation, name=stack_name + "downsample_")
        else:
            nn = conv_bn(nn, nn.shape[-1] * 2, kernel_size=3, strides=2, activation=activation, name=stack_name + "downsample_")
        nn = stack_6(nn, channel, concats=[-1, -3, -5, -6], activation=activation, name=stack_name)

        if id == len(channels) - 1:
            # add SPPCSPC block if it's the last stack
            nn = res_spatial_pyramid_pooling(nn, activation=activation, name=stack_name + "spp_")
        features.append(nn)

    nn = [features[ii] for ii in out_features]
    model = keras.models.Model(inputs, nn, name=model_name)
    return model


""" path aggregation fpn, using `stack_6` instead of csp_stack from yolor """


def upsample_merge(inputs, activation="swish", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    upsample = conv_bn(inputs[-1], inputs[0].shape[-1], activation=activation, name=name + "up_")

    # inputs[0] = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=name + "up")(fpn_out)
    inputs[-1] = tf.image.resize(upsample, tf.shape(inputs[0])[1:-1], method="nearest")
    nn = tf.concat(inputs, axis=-1)
    concats, mid_ratio, out_ratio = [-1, -2, -3, -4, -5, -6], 0.5, 1
    nn = stack_6(nn, nn.shape[-1] // 2, concats=concats, mid_ratio=mid_ratio, out_ratio=out_ratio, activation=activation, name=name)
    return nn


def downsample_merge(inputs, activation="swish", name=""):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    inputs[0] = csp_conv_downsample(inputs[0], ratio=1, activation=activation, name=name)
    nn = tf.concat(inputs, axis=-1)
    concats, mid_ratio, out_ratio = [-1, -2, -3, -4, -5, -6], 0.5, 1
    nn = stack_6(nn, nn.shape[-1] // 2, concats=concats, mid_ratio=mid_ratio, out_ratio=out_ratio, activation=activation, name=name)
    return nn


def path_aggregation_fpn(features, activation="swish", name=""):
    # p5 ─────┬───> pan_out0
    #         ↓         ↑
    # p4 ─> p4p5 ─> pan_out1
    #         ↓         ↑
    # p3 ─> pan_out2 ───┘
    # features: [p3, p4, p5]
    upsamples = [features[-1]]
    p_name = "p{}_".format(len(features) + 2)
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, ii in enumerate(features[:-1][::-1]):
        cur_p_name = "p{}".format(len(features) + 1 - id)
        nn = conv_bn(ii, ii.shape[-1] // 4, kernel_size=1, activation=activation, name=name + cur_p_name + "_down_")
        p_name = cur_p_name + p_name
        nn = upsample_merge([nn, upsamples[-1]], activation=activation, name=name + p_name)
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = name + "c3n{}_".format(id + 3)
        nn = downsample_merge([downsamples[-1], ii], activation=activation, name=cur_name)
        downsamples.append(nn)
    return downsamples


""" YOLOV7Head, using Reparam Conv block """


def yolov7_head_single(inputs, filters, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", name=""):
    # OREPA_3x3_RepConv
    rep_conv_3 = conv_bn(inputs, filters, 3, activation=None, name=name + "3x3_")
    rep_conv_1 = conv_bn(inputs, filters, 1, activation=None, name=name + "1x1_")
    nn = keras.layers.Add()([rep_conv_3, rep_conv_1])
    nn = activation_by_name(nn, activation=activation, name=name)

    ouput_classes = num_classes + (5 if use_object_scores else 4)  # num_anchors = 3, num_anchors * (80 + 5) = 255
    nn = keras.layers.Conv2D(ouput_classes * num_anchors, kernel_size=1, name=name + "2_conv")(nn)
    return nn
    # return keras.layers.Reshape([-1, ouput_classes], name=name + "output_reshape")(nn)


def yolov7_head(inputs, num_classes=80, num_anchors=3, use_object_scores=True, activation="swish", classifier_activation="sigmoid", name=""):
    outputs = []
    for id, input in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        filters = int(input.shape[-1] * 2)
        out = yolov7_head_single(input, filters, num_classes, num_anchors, use_object_scores, activation=activation, name=cur_name)
        outputs.append(out)
    return outputs
    # outputs = tf.concat(outputs, axis=1)
    # return activation_by_name(outputs, classifier_activation, name="classifier_")


""" YOLOV7 models, almost same with yolor """


def YOLOV7(
    backbone=None,
    csp_channels=[64, 128, 256, 256],
    stem_width=-1,  # -1 means using csp_channels[0] // 2
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
    model_name="yolov7",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
    rescale_mode="raw01",  # For decode predictions, raw01 means input value in range [0, 1].
    kwargs=None,  # Not using, recieving parameter
):
    if backbone is None:
        backbone = Yolov7Backbone(csp_channels, stem_width, features_pick, input_shape=input_shape, activation=activation, model_name="backbone")
        features = backbone.outputs
    else:
        if isinstance(features_pick[0], str):
            features = [backbone.get_layer(layer_name) for layer_name in features_pick]
        else:
            features = model_surgery.get_pyramide_feature_layers(backbone)
            features = [features[id] for id in features_pick]
        print(">>>> features:", {ii.name: ii.output_shape for ii in features})
        features = [ii.output for ii in features]

    backbone.trainable = False if freeze_backbone else True
    use_object_scores, num_anchors, anchor_scale = anchors_func.get_anchors_mode_parameters(anchors_mode, use_object_scores, num_anchors, anchor_scale)
    inputs = backbone.inputs[0]

    fpn_features = path_aggregation_fpn(features, activation=activation, name="pafpn_")
    outputs = yolov7_head(fpn_features, num_classes, num_anchors, use_object_scores, activation, classifier_activation, name="head_")
    outputs = keras.layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = keras.models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolov7", pretrained)

    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    post_process = eval_func.DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchors_mode, use_object_scores, anchor_scale)
    # add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


def YOLOV7_CSP(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, classifier_activation="sigmoid", pretrained="coco", **kwargs):
    return YOLOV7(**locals(), model_name=kwargs.pop("model_name", "yolov7_csp"), **kwargs)
