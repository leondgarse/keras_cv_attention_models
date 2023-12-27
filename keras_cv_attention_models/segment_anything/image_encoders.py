import keras_cv_attention_models
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, layer_norm
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6
BATCH_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "mobile_sam_5m_image_encoder": {"sam": {1024: "d9e48e1b5109b8f677625454a5f9c257"}},
    "tiny_sam_5m_image_encoder": {"sam": {1024: "ae58fa89388f5e1d414e86c33b21a71a"}},
    "efficientvit_sam_l0_image_encoder": {"sam": {1024: "d91f40cf7f46b375a859bef4b2c87bdb"}},
}

IMAGE_ENCODERS = {
    "tinyvit5m": "ImageEncoder_TinyViT_5M",
    "mobilesam": "ImageEncoder_TinyViT_5M",
    "mobilesam5m": "ImageEncoder_TinyViT_5M",
    "efficientvitl0": "ImageEncoder_EfficientViT_L0",
}


def get_image_encoder(model="mobile_sam_5m", input_shape=(1024, 1024, 3), embed_dims=256, pretrained="sam", name="image_encoder"):
    if not isinstance(model, str):
        return model

    simple_model_name = model.lower().replace("-", "").replace("_", "")
    model_name = IMAGE_ENCODERS[simple_model_name]
    print(">>>> image_encoder model name:", model_name)
    return globals()[model_name](input_shape=input_shape, embed_dims=embed_dims, pretrained=pretrained, name=name)


""" MobileSAM """


def ImageEncoder_TinyViT_5M(input_shape=(1024, 1024, 3), embed_dims=256, pretrained="sam", name="mobile_sam_5m_image_encoder"):
    base_window_ration = input_shape[1] / 32 / 7  # keep window_size=[7, 7, 14, 7]
    window_ratios = [base_window_ration * 8, base_window_ration * 4, base_window_ration, base_window_ration * 2]
    backbone = keras_cv_attention_models.models.TinyViT_5M(
        input_shape=input_shape, window_ratios=window_ratios, strides=[1, 2, 2, 1], num_classes=0, pretrained=None
    )
    inputs = backbone.inputs[0]
    nn = backbone.outputs[0]
    nn = conv2d_no_bias(nn, embed_dims, kernel_size=1, use_bias=False, name="neck_1_")
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="neck_1_")
    nn = conv2d_no_bias(nn, embed_dims, kernel_size=3, padding="SAME", use_bias=False, name="neck_2_")
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="neck_2_")

    model = models.Model(inputs, nn, name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


""" EfficientViT_L """


def ImageEncoder_EfficientViT_L0(
    input_shape=(512, 512, 3), embed_dims=256, middle_depth=4, activation="gelu/app", pretrained="sam", name="efficientvit_sam_l0_image_encoder"
):
    from keras_cv_attention_models.efficientvit import efficientvit_b

    """ Backbone """
    keras_cv_attention_models.efficientvit_b.LAYER_NORM_EPSILON = LAYER_NORM_EPSILON  # original is 1e-5
    keras_cv_attention_models.efficientvit_b.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON  # original is 1e-5
    backbone = keras_cv_attention_models.efficientvit_b.EfficientViT_L0(
        input_shape=input_shape, num_classes=0, output_filters=[0, 0], pretrained=None, activation=activation
    )
    inputs = backbone.inputs[0]
    features = keras_cv_attention_models.model_surgery.get_pyramide_feature_layers(backbone)

    """ In """
    merged_features = []
    # middle_size = features[-2].output.shape[1:-1] if image_data_format() == "channels_last" else features[-2].output.shape[2:]
    for id, nn in enumerate([ii.output for ii in features[-3:]]):
        nn = conv2d_no_bias(nn, embed_dims, name="features_{}_".format(id + 1))
        nn = batchnorm_with_activation(nn, activation=None, epsilon=BATCH_NORM_EPSILON, name="features_{}_".format(id + 1))
        cur_height, cur_width = nn.shape[1:-1] if image_data_format() == "channels_last" else nn.shape[2:]
        if cur_height != 64 or cur_width != 64:
            nn = functional.resize(nn, [64, 64], method="bicubic", antialias=False)  # shape fixed as [64, 64]
        merged_features.append(nn)

    """ Middle """
    middle = layers.Add()(merged_features)
    for id in range(middle_depth):
        nn = conv2d_no_bias(middle, embed_dims, 3, padding="same", name="middle_{}_expand_".format(id + 1))
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, name="middle_{}_expand_".format(id + 1))
        nn = conv2d_no_bias(nn, embed_dims, 1, strides=1, padding="same", name="middle_{}_pw_".format(id + 1))
        nn = batchnorm_with_activation(nn, activation=None, epsilon=BATCH_NORM_EPSILON, zero_gamma=True, name="middle_{}_pw_".format(id + 1))
        middle = layers.Add(name="middle_{}_output_".format(id + 1))([middle, nn])

    """ Out """
    out = conv2d_no_bias(middle, embed_dims, kernel_size=1, use_bias=True, name="out_")
    out = layer_norm(out, epsilon=LAYER_NORM_EPSILON, name="out_")

    model = models.Model(inputs, out, name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model
