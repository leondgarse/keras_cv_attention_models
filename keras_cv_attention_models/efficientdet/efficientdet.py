import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models.coco.data import get_anchors, decode_bboxes
from keras_cv_attention_models import efficientnet
from keras_cv_attention_models.attention_layers import activation_by_name, add_pre_post_process
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_EPSILON = 1e-3
PRETRAINED_DICT = {
    "efficientdet_d0": {"coco": {512: "d5ba1c9ffd3627571bb00091e3fde28e"}},
    "efficientdet_d1": {"coco": {640: "c34d8603ea1eefbd7f10415a5b51d824"}},
    "efficientdet_d2": {"coco": {768: "bfd7d19cd5a770e478b51c09e2182dc7"}},
    "efficientdet_d3": {"coco": {896: "39d7b5ac515e073123686c200573b7be"}},
    "efficientdet_d4": {"coco": {1024: "b80d45757f139ef27bf5cf8de0746222"}},
    "efficientdet_d5": {"coco": {1280: "e7bc8987e725d8779684da580a42efa9"}},
    "efficientdet_d6": {"coco": {1280: "9302dc59b8ade929ea68bd68fc71c9f3"}},
    "efficientdet_d7": {"coco": {1536: "6d615faf95a4891aec12392f17155950"}},
    "efficientdet_d7x": {"coco": {1536: "a1e4b4e8e488eeabee14c7801028984d"}},
}


@tf.keras.utils.register_keras_serializable(package="efficientdet")
class ReluWeightedSum(keras.layers.Layer):
    def __init__(self, initializer="ones", epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.initializer, self.epsilon = initializer, epsilon

    def build(self, input_shape):
        self.total = len(input_shape)
        self.gain = self.add_weight(name="gain", shape=(self.total,), initializer=self.initializer, dtype=self.dtype, trainable=True)
        self.__epsilon__ = tf.cast(self.epsilon, self._compute_dtype)

    def call(self, inputs):
        gain = tf.nn.relu(self.gain)
        gain = gain / (tf.reduce_sum(gain) + self.__epsilon__)
        return tf.reduce_sum([inputs[id] * gain[id] for id in range(self.total)], axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"initializer": self.initializer, "epsilon": self.epsilon})
        return base_config


def align_feature_channel(inputs, output_channel, name=""):
    # print(f">>>> align_feature_channel: {name = }, {inputs.shape = }, {output_channel = }")
    nn = keras.layers.Conv2D(output_channel, kernel_size=1, name=name + "channel_conv")(inputs)
    nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "channel_bn")(nn)
    return nn


def align_feature_blocks_2(inputs, downsample=True, interpolation="nearest", name=""):
    # print(">>>>", "downsample:" if downsample else "upsample:", f"{name = }, {inputs.shape = }, {downsample = }")
    if downsample:
        nn = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="SAME", name=name + "max_down")(inputs)
    else:
        nn = keras.layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "up")(inputs)
    return nn


def resample_fusion(inputs, output_channel, use_weighted_sum=True, downsample=True, interpolation="nearest", activation="swish", name=""):
    if inputs[0].shape[-1] != output_channel:
        inputs[0] = align_feature_channel(inputs[0], output_channel, name=name)
    # downsample = inputs[-1].shape[1] > inputs[0].shape[1]
    inputs[-1] = align_feature_blocks_2(inputs[-1], downsample=downsample, interpolation=interpolation, name=name)

    if use_weighted_sum:
        nn = ReluWeightedSum(name=name + "wsm")(inputs)
    else:
        nn = keras.layers.Add(name=name + "sum")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    nn = keras.layers.SeparableConv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=name + "sepconv")(nn)
    nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "bn")(nn)
    return nn


def bi_fpn(features, output_channel, use_weighted_sum=True, activation="swish", name=""):
    # print(f">>>> bi_fpn: {[ii.shape for ii in features] = }")
    # features: [p3, p4, p5, p6, p7]
    up_features = [features[-1]]
    for id, feature in enumerate(features[:-1][::-1]):
        cur_name = name + "p{}_up_".format(len(features) - id + 1)
        up_feature = resample_fusion([feature, up_features[-1]], output_channel, use_weighted_sum, downsample=False, activation=activation, name=cur_name)
        up_features.append(up_feature)

    # up_features: [p7, p6_up, p5_up, p4_up, p3_up]
    out_features = [up_features[-1]]  # [p3_up]
    up_features = up_features[1:-1][::-1]  # [p4_up, p5_up, p6_up]
    for id, feature in enumerate(features[1:]):
        cur_name = name + "p{}_out_".format(len(features) - 1 + id)
        fusion_feature = [feature, out_features[-1]] if id == len(up_features) else [feature, up_features[id], out_features[-1]]
        out_feature = resample_fusion(fusion_feature, output_channel, use_weighted_sum, downsample=True, activation=activation, name=cur_name)
        out_features.append(out_feature)

    # out_features: [p3_up, p4_out, p5_out, p6_out, p7_out]
    return out_features


def detector_head(features, output_channel, repeats, num_classes=80, num_anchors=9, activation="swish", head_activation="sigmoid", name=""):
    names = [name + "{}_sepconv".format(id + 1) for id in range(repeats)]
    sep_convs = [keras.layers.SeparableConv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=names[id]) for id in range(repeats)]
    outputs = []
    for feature_id, feature in enumerate(features):
        nn = feature
        for id in range(repeats):
            nn = sep_convs[id](nn)
            cur_name = name + "{}_{}_bn".format(id + 1, feature_id + 1)
            nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=cur_name)(nn)
            nn = activation_by_name(nn, activation, name=cur_name + "{}_".format(id + 1))
        outputs.append(nn)

    if num_classes > 0:
        header_conv = keras.layers.SeparableConv2D(num_classes * num_anchors, kernel_size=3, padding="SAME", use_bias=True, name=name + "head")
        outputs = [header_conv(ii) for ii in outputs]
        outputs = [keras.layers.Reshape([-1, num_classes])(ii) for ii in outputs]
        outputs = tf.concat(outputs, axis=1)
        outputs = activation_by_name(outputs, head_activation, name=name + "output_")
    return outputs


def EfficientDet(
    backbone,
    features_pick=[-3, -2, -1],  # The last 3 ones, or specific layer_names / feature_indexes
    additional_features=2,  # Add p5->p6, p6->p7
    fpn_depth=3,
    head_depth=3,
    num_channels=64,
    use_weighted_sum=True,
    num_anchors=9,
    num_classes=90,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    anchor_scale=4,  # Init anchors for model prediction
    pyramid_levels=[3, 7],  # Init anchors for model prediction
    pretrained="coco",
    model_name=None,
    kwargs=None,  # Not using, recieving parameter
    input_shape=None,  # Not using, recieving parameter
):
    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True

    if isinstance(features_pick[0], str):
        fpn_features = [backbone.get_layer(layer_name) for layer_name in features_pick]
    else:
        features = model_surgery.get_pyramide_feture_layers(backbone)
        fpn_features = [features[id] for id in features_pick]
    print(">>>> features:", {ii.name: ii.output_shape for ii in fpn_features})
    fpn_features = [ii.output for ii in fpn_features]

    # Build additional input features that are not from backbone.
    for id in range(additional_features):
        cur_name = "p{}_p{}_".format(id + 5, id + 6)
        additional_feature = fpn_features[-1]
        if fpn_features[-1].shape[-1] != num_channels:
            additional_feature = align_feature_channel(additional_feature, num_channels, name=cur_name)
        additional_feature = align_feature_blocks_2(additional_feature, downsample=True, name=cur_name)
        fpn_features.append(additional_feature)

    for id in range(fpn_depth):
        fpn_features = bi_fpn(fpn_features, num_channels, use_weighted_sum, activation=activation, name="biFPN_{}_".format(id + 1))

    bbox_regressor = detector_head(fpn_features, num_channels, head_depth, 4, num_anchors, activation, head_activation=None, name="regressor_")
    if num_classes > 0:
        classifier = detector_head(fpn_features, num_channels, head_depth, num_classes, num_anchors, activation, classifier_activation, name="classifier_")
        outputs = tf.concat([bbox_regressor, classifier], axis=-1)
    else:
        outputs = bbox_regressor
    # outputs = keras.layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)

    model_name = model_name or backbone.name + "_det"
    model = keras.models.Model(inputs=backbone.inputs[0], outputs=outputs, name=model_name)
    add_pre_post_process(model, rescale_mode="torch", post_process=DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchor_scale))
    reload_model_weights(model, PRETRAINED_DICT, "efficientdet", pretrained)
    return model


class DecodePredictions:
    def __init__(self, input_shape=(512, 512, 3), pyramid_levels=[3, 7], anchor_scale=4, **kwargs):
        self.anchor_scale, self.kwargs = anchor_scale, kwargs
        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        if input_shape[0] is not None:
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None

    def __init_anchor__(self, input_shape):
        self.anchors = get_anchors(input_shape=input_shape, pyramid_levels=self.pyramid_levels, anchor_scale=self.anchor_scale, **self.kwargs)

    def __decode_one__(self, preds, score_threshold=0.3, iou_threshold=0.2, max_output_size=15, input_shape=None):
        if input_shape is not None:
            self.__init_anchor__(input_shape)
        preds_decode = decode_bboxes(preds, self.anchors).numpy()
        cc = preds_decode[tf.reduce_max(preds_decode[:, 4:], -1) > score_threshold]
        rr = tf.image.non_max_suppression(cc[:, :4], cc[:, 4:].max(-1), max_output_size=max_output_size, iou_threshold=iou_threshold)
        cc_nms = tf.gather(cc, rr).numpy()
        bboxes, labels, confidences = cc_nms[:, :4], cc_nms[:, 4:].argmax(-1), cc_nms[:, 4:].max(-1)
        return bboxes, labels, confidences

    def __call__(self, preds, score_threshold=0.3, iou_threshold=0.3, max_output_size=15, input_shape=None):
        preds = preds if len(preds.shape) == 3 else [preds]
        return [self.__decode_one__(pred, score_threshold, iou_threshold, max_output_size, input_shape) for pred in preds]


def EfficientDetD0(input_shape=(512, 512, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B0(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block1_output", "stack_4_block2_output", "stack_6_block0_output"]
    return EfficientDet(**locals(), fpn_depth=3, head_depth=3, num_channels=64, model_name="efficientdet_d0", **kwargs)


def EfficientDetD1(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B1(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block1_output"]
    return EfficientDet(**locals(), fpn_depth=4, head_depth=3, num_channels=88, model_name="efficientdet_d1", **kwargs)


def EfficientDetD2(input_shape=(768, 768, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B2(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block1_output"]
    return EfficientDet(**locals(), fpn_depth=5, head_depth=3, num_channels=112, model_name="efficientdet_d2", **kwargs)


def EfficientDetD3(input_shape=(896, 896, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B3(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block2_output", "stack_4_block4_output", "stack_6_block1_output"]
    return EfficientDet(**locals(), fpn_depth=6, head_depth=4, num_channels=160, model_name="efficientdet_d3", **kwargs)


def EfficientDetD4(input_shape=(1024, 1024, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B4(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block3_output", "stack_4_block5_output", "stack_6_block1_output"]
    return EfficientDet(**locals(), fpn_depth=7, head_depth=4, num_channels=224, model_name="efficientdet_d4", **kwargs)


def EfficientDetD5(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B5(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block4_output", "stack_4_block6_output", "stack_6_block2_output"]
    return EfficientDet(**locals(), fpn_depth=7, head_depth=4, num_channels=288, model_name="efficientdet_d5", **kwargs)


def EfficientDetD6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B6(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block5_output", "stack_4_block7_output", "stack_6_block2_output"]
    return EfficientDet(**locals(), fpn_depth=8, head_depth=5, num_channels=384, use_weighted_sum=False, model_name="efficientdet_d6", **kwargs)


def EfficientDetD7(input_shape=(1536, 1536, 3), freeze_backbone=False, num_classes=90, anchor_scale=5, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B6(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block5_output", "stack_4_block7_output", "stack_6_block2_output"]
    return EfficientDet(**locals(), fpn_depth=8, head_depth=5, num_channels=384, use_weighted_sum=False, model_name="efficientdet_d7", **kwargs)


def EfficientDetD7X(input_shape=(1536, 1536, 3), freeze_backbone=False, num_classes=90, pretrained="coco", **kwargs):
    backbone = efficientnet.EfficientNetV1B7(input_shape=input_shape, num_classes=0, output_conv_filter=0, pretrained=None)
    features_pick = ["stack_2_block6_output", "stack_4_block9_output", "stack_6_block3_output"]
    fpn_depth = 8
    head_depth = 5
    num_channels = 384
    use_weighted_sum = False
    additional_features = 3
    pyramid_levels = [3, 8]
    return EfficientDet(**locals(), model_name="efficientdet_d7x", **kwargs)
