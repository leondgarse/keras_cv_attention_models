from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "vanillanet_10_deploy": {"imagenet": "554b5bbd83141d83fe81d8a86e6d197f"},
    "vanillanet_10": {"imagenet": "ad0e864e059c0681c8a828819f30e8be"},
    "vanillanet_5_deploy": {"imagenet": "39df8736f61ef9536d33be025849a1ac"},
    "vanillanet_5": {"imagenet": "1dcbdec8d9c9a0b7c1a526528bea2e54"},
    "vanillanet_6_deploy": {"imagenet": "8d53cc4b8a9e82347b63fc839766a3bd"},
    "vanillanet_6": {"imagenet": "68d3c1c57711f806ced2cbc410c5d76f"},
    "vanillanet_7_deploy": {"imagenet": "cefc8bf1ec4507a2abe9d7de69245608"},
    "vanillanet_7": {"imagenet": "f2f168fcc59fc69db9a67dee6c23afad"},
    "vanillanet_8_deploy": {"imagenet": "b9deae326d5c0888c9c4a365ea1e99ad"},
    "vanillanet_8": {"imagenet": "20090c6bdef74e86754fcea96ea7ffc9"},
    "vanillanet_9_deploy": {"imagenet": "ef2b73dee1a0ce45865178ae0fcd4656"},
    "vanillanet_9": {"imagenet": "922fb62b1cb81786c14845b020f70082"},
}


BATCH_NORM_EPSILON = 1e-6


def activation_depthwise_conv_bn(inputs, kernel_size=7, deploy=False, activation="relu", name=""):
    nn = activation_by_name(inputs, activation=activation, name=name)
    nn = depthwise_conv2d_no_bias(nn, kernel_size=kernel_size, use_bias=deploy, padding="same", name=name)
    if not deploy:
        nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name=name)
    return nn


def VanillaNet(
    out_channels=[1024, 2048, 4096],
    strides=[2, 2, 2],
    stem_width=512,
    leaky_relu_alpha=1.0,
    input_shape=(224, 224, 3),
    deploy=False,
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="vanillanet",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    inner_activation = "leaky_relu/{}".format(leaky_relu_alpha)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=4, strides=4, use_bias=True, name="stem_1_")
    if not deploy:
        nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=inner_activation, name="stem_1_")

        nn = conv2d_no_bias(nn, stem_width, kernel_size=1, strides=1, use_bias=True, name="stem_2_")
        nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name="stem_2_")
    nn = activation_depthwise_conv_bn(nn, deploy=deploy, name="stem_2_act_")

    """ stages """
    for stack_id, (stride, out_channel) in enumerate(zip(strides, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if not deploy:
            input_channel = nn.shape[-1 if image_data_format() == "channels_last" else 1]
            nn = conv2d_no_bias(nn, input_channel, kernel_size=1, use_bias=True, name=stack_name + "1_")
            nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=inner_activation, name=stack_name + "1_")
        nn = conv2d_no_bias(nn, out_channel, kernel_size=1, use_bias=True, name=stack_name + ("1_" if deploy else "2_"))
        if not deploy:
            nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=None, name=stack_name + "2_")

        if stride > 1:
            nn = layers.MaxPool2D(stride, name=stack_name + "down")(nn)
        nn = activation_depthwise_conv_bn(nn, deploy=deploy, name=stack_name + "act_")

    """ output """
    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        nn = conv2d_no_bias(nn, num_classes, kernel_size=1, use_bias=True, name="pre_head_")
        if not deploy:
            nn = batchnorm_with_activation(nn, epsilon=BATCH_NORM_EPSILON, activation=inner_activation, name="pre_head_")
            nn = conv2d_no_bias(nn, num_classes, kernel_size=1, use_bias=True, name="head_")
        if classifier_activation is not None:
            nn = activation_by_name(nn, activation=classifier_activation, name="head_")
        nn = layers.Flatten(dtype="float32", name="head")(nn)

    model = models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "vanillanet", pretrained)

    model.set_leaky_relu_alpha = lambda alpha: set_leaky_relu_alpha(model, alpha)
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    add_pre_post_process(model, rescale_mode="torch")
    return model


def set_leaky_relu_alpha(model, alpha):
    for ii in model.layers:
        if isinstance(ii, layers.LeakyReLU):
            print("set alpha for {}, {} -> {}".format(ii.name, ii.alpha, alpha))
            ii.alpha = alpha


def switch_to_deploy(model):
    from keras_cv_attention_models import model_surgery

    new_model = model_surgery.convert_to_fused_conv_bn_model(model)
    remove_layer_condition = lambda layer: layer["class_name"] == "LeakyReLU" and layer["config"]["alpha"] == 1
    new_model = model_surgery.remove_layer_single_input(new_model, remove_layer_condition=remove_layer_condition)
    new_model = model_surgery.fuse_sequential_conv_strict(new_model)
    add_pre_post_process(new_model, rescale_mode=model.preprocess_input.rescale_mode, post_process=model.decode_predictions)
    return new_model


@register_model
def VanillaNet5(input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return VanillaNet(**locals(), model_name="vanillanet_5" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet6(input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [1024, 2048, 4096, 4096]
    strides = [2, 2, 2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_6" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet7(input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [512, 1024, 2048, 4096, 4096]
    strides = [1, 2, 2, 2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_7" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet8(input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [512, 1024, 2048, 2048, 4096, 4096]
    strides = [1, 2, 2, 1, 2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_8" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet9(input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [512, 1024] + [2048] * 3 + [4096, 4096]
    strides = [1, 2, 2] + [1] * 2 + [2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_9" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet10(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    out_channels = [512, 1024] + [2048] * 4 + [4096, 4096]
    strides = [1, 2, 2] + [1] * 3 + [2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_10" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet11(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    out_channels = [512, 1024] + [2048] * 5 + [4096, 4096]
    strides = [1, 2, 2] + [1] * 4 + [2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_11" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet12(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    out_channels = [512, 1024] + [2048] * 6 + [4096, 4096]
    strides = [1, 2, 2] + [1] * 5 + [2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_12" + ("_deploy" if deploy else ""), **kwargs)


@register_model
def VanillaNet13(
    input_shape=(224, 224, 3), num_classes=1000, deploy=False, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs
):
    out_channels = [512, 1024] + [2048] * 7 + [4096, 4096]
    strides = [1, 2, 2] + [1] * 6 + [2, 1]
    return VanillaNet(**locals(), model_name="vanillanet_13" + ("_deploy" if deploy else ""), **kwargs)
