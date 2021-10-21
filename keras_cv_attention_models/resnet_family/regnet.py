from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights

def RegNetZB(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    hidden_channel_ratio = [[2, 3], [1.5] + [3] * 5, [1.5] + [3] * 11, [2, 3]]
    use_block_output_activation = False
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    se_ratio = 0.25
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    model = AotNet(**locals(), model_name="regnetz_b", **kwargs)
    return model
