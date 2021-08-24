from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "resnet50d": {"imagenet": "deca3680b88300904a09d450e9a8c526"},
    "resnet101d": {"imagenet": "201128b7bf68371399fe134c5e07f3db"},
    "resnet152d": {"imagenet": "1e1823e0c2cf0f7031bc7abc0c9c97ba"},
    "resnet200d": {"imagenet": "39d9050953e8d4fe9c672620542dd24d"},
}

def ResNetD(num_blocks, input_shape=(224, 224, 3), pretrained="imagenet", deep_stem=True, stem_width=32, strides=2, avg_pool_down=True, **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    model = AotNet(num_blocks, input_shape=input_shape, deep_stem=deep_stem, stem_width=stem_width, strides=strides, avg_pool_down=avg_pool_down, **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", input_shape=input_shape, pretrained=pretrained)
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
