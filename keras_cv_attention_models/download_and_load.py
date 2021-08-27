import os
from tensorflow import keras

def reload_model_weights(model, pretrained_dict, sub_release, input_shape=(224, 224, 3), pretrained="imagenet"):
    if model.name not in pretrained_dict or pretrained not in pretrained_dict[model.name]:
        print(">>>> No pretraind available, model will be randomly initialized")
        return None

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.h5"
    url = pre_url.format(sub_release, model.name, pretrained)
    file_name = os.path.basename(url)
    file_hash = pretrained_dict[model.name][pretrained]
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return None
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        return pretrained_model
