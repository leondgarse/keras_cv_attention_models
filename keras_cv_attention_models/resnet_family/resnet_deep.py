from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "resnet50d": {"imagenet": "1b71933a82b058ba1e605ee5c01f64b2"},
    "resnet101d": {"imagenet": "79b075be5cf222cff2bced7a5a117623"},
    "resnet152d": {"imagenet": "0a15299b9abe1fee3ae06d9a59d13a3f"},
    "resnet200d": {"imagenet": "b5961494e0072c342b838c77ef52ddc5"},
}


def ResNetD(num_blocks, input_shape=(224, 224, 3), pretrained="imagenet", stem_type="deep", strides=2, shortcut_type="avg", **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    model = AotNet(num_blocks, input_shape=input_shape, stem_type=stem_type, strides=strides, shortcut_type=shortcut_type, **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def ResNet50D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    return ResNetD(**locals(), model_name="resnet50d", **kwargs)


def ResNet101D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    return ResNetD(**locals(), model_name="resnet101d", **kwargs)


def ResNet152D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 8, 36, 3]
    return ResNetD(**locals(), model_name="resnet152d", **kwargs)


def ResNet200D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 24, 36, 3]
    return ResNetD(**locals(), model_name="resnet200d", **kwargs)
