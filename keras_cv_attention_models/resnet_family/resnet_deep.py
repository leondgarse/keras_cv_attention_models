from keras_cv_attention_models.aotnet import AotNet
import os

def ResNetD(num_blocks, input_shape=(224, 224, 3), pretrained="imagenet", deep_stem=True, stem_width=32, strides=2, **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    model = AotNet(num_blocks, input_shape=input_shape, deep_stem=deep_stem, stem_width=stem_width, strides=strides, **kwargs)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    pretrained_dd = {
        "resnet50d": ["imagenet"],
    }
    if model.name not in pretrained_dd or pretrained not in pretrained_dd[model.name]:
        print(">>>> No pretraind available, model will be randomly initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/{}_{}.h5"
    url = pre_url.format(model.name, pretrained)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


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
