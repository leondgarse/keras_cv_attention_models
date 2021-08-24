from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights


PRETRAINED_DICT = {
    "resnext50": {"imagenet": "cf65d988c38ba0335c97a046288b91f4", "swsl": "f1cf0cc3c49bb50e6949c50fcce3db8f"},
    "resnext101": {"imagenet": "1e58c0ecc31184bd6bfe4d6b568f4325", "swsl": "c2fe8eefcf9a55e0254d2b13055a4cbc"},
    "resnext101w": {"imagenet": "9a1b92145aeb922695c29a0f02b52188", "swsl": "58b7cf4a72b03171f50ed19789b20f3d"},
    "resnext50d": {"imagenet": "a7b2433b7bee7029fce11ba3fabf3fb9"},
}

def ResNeXt(num_blocks, input_shape=(224, 224, 3), pretrained="imagenet", strides=2, attn_types="groups_conv", **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    model = AotNet(num_blocks, input_shape=input_shape, strides=strides, attn_types=attn_types, **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", input_shape=input_shape, pretrained=pretrained)
    return model


def ResNeXt50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    out_channels=[128, 256, 512, 1024]
    expansion = 2
    avg_pool_down = False
    return ResNeXt(**locals(), model_name="resnext50", **kwargs)


def ResNeXt101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    out_channels=[128, 256, 512, 1024]
    expansion = 2
    avg_pool_down = False
    return ResNeXt(**locals(), model_name="resnext101", **kwargs)


def ResNeXt50D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    out_channels=[128, 256, 512, 1024]
    expansion = 2
    deep_stem = True
    stem_width = 32
    avg_pool_down = True
    return ResNeXt(**locals(), model_name="resnext50d", **kwargs)


def ResNeXt101W(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    out_channels = [256, 512, 1024, 2048]
    expansion = 1
    avg_pool_down = False
    return ResNeXt(**locals(), model_name="resnext101w", **kwargs)
