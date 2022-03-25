import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    add_pre_post_process,
)
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco.eval_func import DecodePredictions

PRETRAINED_DICT = {
    "yolox_nano": {"coco": "7c97d60d4cc9d54321176f844acee627"},
    "yolox_tiny": {"coco": "f9b51ff24290090c86a10a45f811140b"},
    "yolox_s": {"coco": "a989f5a808ddc4a8242157a6a3e64977"},
    "yolox_m": {"coco": "5c2333d2f12b2f48e3ec8555b29d242f"},
    "yolox_l": {"coco": "a07c48994b7a67dba421025ef39b858b"},
    "yolox_x": {"coco": "de9741d3f67f50c54856bcae0f07b7ef"},
}


""" CSPDarknet backbone """
BATCH_NORM_EPSILON = 1e-3
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


def csp_stack(inputs, depth, out_channels=-1, expansion=0.5, use_shortcut=True, use_depthwise_conv=False, activation="swish", name=""):
    out_channels = inputs.shape[-1] if out_channels == -1 else out_channels
    hidden_channels = int(out_channels * expansion)
    short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_")
    for id in range(depth):
        block_name = name + "block{}_".format(id + 1)
        deep = csp_block(deep, 1, use_shortcut=use_shortcut, use_depthwise_conv=use_depthwise_conv, activation=activation, name=block_name)

    out = tf.concat([deep, short], axis=-1)
    out = conv_dw_pw_block(out, out_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def spatial_pyramid_pooling(inputs, pool_sizes=(5, 9, 13), activation="swish", name=""):
    input_channels = inputs.shape[-1]
    nn = conv_dw_pw_block(inputs, input_channels // 2, kernel_size=1, activation=activation, name=name + "1_")
    pp = [keras.layers.MaxPooling2D(pool_size=ii, strides=1, padding="SAME")(nn) for ii in pool_sizes]
    nn = tf.concat([nn, *pp], axis=-1)
    nn = conv_dw_pw_block(nn, input_channels, kernel_size=1, activation=activation, name=name + "2_")
    return nn


def focus_stem(inputs, filters, kernel_size=3, strides=1, padding="valid", activation="swish", name=""):
    if padding.lower() == "same":  # Handling odd input_shape
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
        patch_top_left = inputs[:, :-1:2, :-1:2]
        patch_top_right = inputs[:, :-1:2, 1::2]
        patch_bottom_left = inputs[:, 1::2, :-1:2]
        patch_bottom_right = inputs[:, 1::2, 1::2]
    else:
        patch_top_left = inputs[:, ::2, ::2]
        patch_top_right = inputs[:, ::2, 1::2]
        patch_bottom_left = inputs[:, 1::2, ::2]
        patch_bottom_right = inputs[:, 1::2, 1::2]
    nn = tf.concat([patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right], axis=-1)
    nn = conv_dw_pw_block(nn, filters, kernel_size=kernel_size, strides=strides, activation=activation, name=name)
    return nn


def CSPDarknet(width_mul=1, depth_mul=1, out_features=[-3, -2, -1], use_depthwise_conv=False, input_shape=(512, 512, 3), activation="swish", model_name=""):
    base_channels, base_depth = int(width_mul * 64), max(round(depth_mul * 3), 1)
    inputs = keras.layers.Input(input_shape)

    """ Stem """
    nn = focus_stem(inputs, base_channels, activation=activation, name="stem_")
    features = [nn]

    """ dark blocks """
    depthes = [base_depth, base_depth * 3, base_depth * 3, base_depth]
    channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
    use_spps = [False, False, False, True]
    use_shortcuts = [True, True, True, False]
    for id, (channel, depth, use_spp, use_shortcut) in enumerate(zip(channels, depthes, use_spps, use_shortcuts)):
        stack_name = "stack{}_".format(id + 1)
        nn = conv_dw_pw_block(nn, channel, kernel_size=3, strides=2, use_depthwise_conv=use_depthwise_conv, activation=activation, name=stack_name)
        if use_spp:
            nn = spatial_pyramid_pooling(nn, activation=activation, name=stack_name + "spp_")
        # nn = SPPBottleneck(base_channels * 16, base_channels * 16, activation=act)
        nn = csp_stack(nn, depth, use_shortcut=use_shortcut, use_depthwise_conv=use_depthwise_conv, activation=activation, name=stack_name)
        features.append(nn)

    nn = [features[ii] for ii in out_features]
    model = keras.models.Model(inputs, nn, name=model_name)
    return model


""" path aggregation fpn """


def upsample_merge(inputs, csp_depth, use_depthwise_conv=False, activation="swish", name=""):
    # print(f">>>> upsample_merge inputs: {[ii.shape for ii in inputs] = }")
    target_channel = inputs[-1].shape[-1]
    fpn_out = conv_dw_pw_block(inputs[0], target_channel, activation=activation, name=name + "fpn_")

    # inputs[0] = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=name + "up")(fpn_out)
    inputs[0] = tf.image.resize(fpn_out, tf.shape(inputs[-1])[1:-1], method="nearest")
    nn = tf.concat(inputs, axis=-1)
    nn = csp_stack(nn, csp_depth, target_channel, 0.5, False, use_depthwise_conv, activation=activation, name=name)
    return fpn_out, nn


def downsample_merge(inputs, csp_depth, use_depthwise_conv=False, activation="swish", name=""):
    # print(f">>>> downsample_merge inputs: {[ii.shape for ii in inputs] = }")
    inputs[0] = conv_dw_pw_block(inputs[0], inputs[-1].shape[-1], 3, 2, use_depthwise_conv, activation=activation, name=name + "down_")
    nn = tf.concat(inputs, axis=-1)
    nn = csp_stack(nn, csp_depth, nn.shape[-1], 0.5, False, use_depthwise_conv, activation=activation, name=name)
    return nn


def path_aggregation_fpn(features, depth_mul=1, use_depthwise_conv=False, activation="swish", name=""):
    # p5 ─> fpn_out0 ───────────> pan_out0
    #          ↓                     ↑
    # p4 ─> f_out0 ─> fpn_out1 ─> pan_out1
    #                    ↓           ↑
    # p3 ───────────> pan_out2 ──────┘
    csp_depth = max(round(depth_mul * 3), 1)
    p3, p4, p5 = features  # p3: [64, 64, 256], p4: [32, 32, 512], p5: [16, 16, 1024]
    # fpn_out0: [16, 16, 512], f_out0: [32, 32, 512]
    fpn_out0, f_out0 = upsample_merge([p5, p4], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3p4_")
    # fpn_out1: [32, 32, 256], pan_out2: [64, 64, 256]
    fpn_out1, pan_out2 = upsample_merge([f_out0, p3], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3p3_")
    # pan_out1: [32, 32, 512]
    pan_out1 = downsample_merge([pan_out2, fpn_out1], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3n3_")
    # pan_out0: [16, 16, 1024]
    pan_out0 = downsample_merge([pan_out1, fpn_out0], csp_depth, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "c3n4_")
    return [pan_out2, pan_out1, pan_out0]


""" YOLOXHead """


def yolox_head_single(inputs, out_channels, num_classes=80, num_anchors=1, use_depthwise_conv=False, use_object_scores=True, activation="swish", name=""):
    bias_init = tf.constant_initializer(-tf.math.log((1 - 0.01) / 0.01).numpy())

    # stem
    stem = conv_dw_pw_block(inputs, out_channels, activation=activation, name=name + "stem_")

    # cls_convs, cls_preds
    cls_nn = conv_dw_pw_block(stem, out_channels, kernel_size=3, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "cls_1_")
    cls_nn = conv_dw_pw_block(cls_nn, out_channels, kernel_size=3, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "cls_2_")
    cls_out = keras.layers.Conv2D(num_classes * num_anchors, kernel_size=1, bias_initializer=bias_init, name=name + "class_out")(cls_nn)
    cls_out = activation_by_name(cls_out, "sigmoid", name=name + "class_out_")
    cls_out = keras.layers.Reshape([-1, num_classes], name=name + "class_out_reshape")(cls_out)

    # reg_convs, reg_preds
    reg_nn = conv_dw_pw_block(stem, out_channels, kernel_size=3, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "reg_1_")
    reg_nn = conv_dw_pw_block(reg_nn, out_channels, kernel_size=3, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "reg_2_")
    reg_out = keras.layers.Conv2D(4 * num_anchors, kernel_size=1, name=name + "regression_out")(reg_nn)
    reg_out = keras.layers.Reshape([-1, 4], name=name + "regression_out_reshape")(reg_out)

    # obj_preds
    if use_object_scores:
        obj_out = keras.layers.Conv2D(1 * num_anchors, kernel_size=1, bias_initializer=bias_init, name=name + "object_out")(reg_nn)
        obj_out = activation_by_name(obj_out, "sigmoid", name=name + "object_out_")
        obj_out = keras.layers.Reshape([-1, 1], name=name + "object_out_reshape")(obj_out)
        return tf.concat([reg_out, cls_out, obj_out], axis=-1)
    else:
        return tf.concat([reg_out, cls_out], axis=-1)


def yolox_head(inputs, width_mul=1.0, num_classes=80, num_anchors=1, use_depthwise_conv=False, use_object_scores=True, activation="swish", name=""):
    out_channel = int(256 * width_mul)
    outputs = []
    for id, input in enumerate(inputs):
        cur_name = name + "{}_".format(id + 1)
        out = yolox_head_single(input, out_channel, num_classes, num_anchors, use_depthwise_conv, use_object_scores, activation=activation, name=cur_name)
        outputs.append(out)
    # outputs = tf.concat([keras.layers.Reshape([-1, ii.shape[-1]])(ii) for ii in outputs], axis=1)
    outputs = tf.concat(outputs, axis=1)
    return outputs


""" YOLOX models """


def YOLOX(
    backbone=None,
    features_pick=[-3, -2, -1],
    depth_mul=1,
    width_mul=-1,  # -1 means: `min([ii.shape[-1] for ii in features]) / 256` for custom backbones.
    use_depthwise_conv=False,
    use_anchor_free_mode=True,
    use_yolor_anchors_mode=False,
    num_anchors="auto",  # "auto" means 1 if use_anchor_free_mode else 9
    use_object_scores="auto",  # "auto" means same with use_anchor_free_mode
    input_shape=(640, 640, 3),
    num_classes=80,
    activation="swish",
    freeze_backbone=False,
    pretrained=None,
    model_name="yolox",
    pyramid_levels_min=3,  # Init anchors for model prediction.
    anchor_scale="auto",  # Init anchors for model prediction. "auto" means 1 if use_anchor_free_mode else 4
    rescale_mode="raw",  # For decode predictions, raw means input value in range [0, 255].
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
        width_mul = width_mul if width_mul > 0 else min([ii.shape[-1] for ii in features]) / 256
        print(">>>> width_mul:", width_mul)

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True

    use_object_scores = (use_yolor_anchors_mode or use_anchor_free_mode) if use_object_scores == "auto" else use_object_scores
    if num_anchors == "auto":
        num_anchors = 1 if use_anchor_free_mode else (3 if use_yolor_anchors_mode else 9)

    inputs = backbone.inputs[0]
    fpn_features = path_aggregation_fpn(features, depth_mul=depth_mul, use_depthwise_conv=use_depthwise_conv, activation=activation, name="pafpn_")
    outputs = yolox_head(fpn_features, width_mul, num_classes, num_anchors, use_depthwise_conv, use_object_scores, activation=activation, name="head_")
    outputs = keras.layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)
    model = keras.models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "yolox", pretrained)

    # AA = {"aspect_ratios": anchor_aspect_ratios, "num_scales": anchor_num_scales, "anchor_scale": anchor_scale, "grid_zero_start": anchor_grid_zero_start}
    pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
    anchor_scale = (1 if use_anchor_free_mode or use_yolor_anchors_mode else 4) if anchor_scale == "auto" else anchor_scale
    post_process = DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchor_scale, use_anchor_free_mode, use_yolor_anchors_mode, use_object_scores)
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model


def YOLOXNano(input_shape=(416, 416, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=0.33, width_mul=0.25, use_depthwise_conv=True, model_name=kwargs.pop("model_name", "yolox_nano"), **kwargs)


def YOLOXTiny(input_shape=(416, 416, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=0.33, width_mul=0.375, model_name=kwargs.pop("model_name", "yolox_tiny"), **kwargs)


def YOLOXS(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=0.33, width_mul=0.5, model_name=kwargs.pop("model_name", "yolox_s"), **kwargs)


def YOLOXM(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=0.67, width_mul=0.75, model_name=kwargs.pop("model_name", "yolox_m"), **kwargs)


def YOLOXL(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=1.0, width_mul=1.0, model_name=kwargs.pop("model_name", "yolox_l"), **kwargs)


def YOLOXX(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, backbone=None, activation="swish", pretrained="coco", **kwargs):
    return YOLOX(**locals(), depth_mul=1.33, width_mul=1.25, model_name=kwargs.pop("model_name", "yolox_x"), **kwargs)
