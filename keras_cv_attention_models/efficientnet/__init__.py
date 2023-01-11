from keras_cv_attention_models.efficientnet.efficientnet_v2 import (
    EfficientNetV2,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2T,
    EfficientNetV2T_GC,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    EfficientNetV2XL,
)
from keras_cv_attention_models.efficientnet.efficientnet_v1 import (
    EfficientNetV1,
    EfficientNetV1B0,
    EfficientNetV1B1,
    EfficientNetV1B2,
    EfficientNetV1B3,
    EfficientNetV1B4,
    EfficientNetV1B5,
    EfficientNetV1B6,
    EfficientNetV1B7,
    EfficientNetV1L2,
    EfficientNetV1Lite0,
    EfficientNetV1Lite1,
    EfficientNetV1Lite2,
    EfficientNetV1Lite3,
    EfficientNetV1Lite4,
)

__v2_head_doc__ = """
Keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2).
Paper [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels. Set `(None, None, 3)` for dynamic.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  dropout: dropout rate if top layers is included.
  first_strides: is used in the first Conv2D layer.
  drop_connect_rate: is used for: Deep Networks with Stochastic Depth.
      arxiv: https://arxiv.org/abs/1603.09382.
      If not `0`, will add a `Dropout` layer for each deep branch changes
      from `0 --> drop_connect_rate` for `top --> bottom` layers. Or `0` to disable.
      Default `0` for `EfficientNetV2`, `0.2` for `EfficientNetV1`.
  classifier_activation: A `str` or callable.
      The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  include_preprocessing: Boolean value if add preprocessing `Rescale` / `Normalization` after `Input`, will expect input value in range `[0, 255]`.
      Note that different preprocessing will applied for different `rescale_mode`. It depends on `pretained` and `model_type`.
      - For all V1 models, `rescale_mode`s are "torch".
      - For "21k" pretrained V2 models, `rescale_mode`s are all "tf"
      - For "imagenet" pretrained V2 models, "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf".
      Default `False`.
  pretrained: value in {pretrained}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/efficientnetv2/`.

Returns:
    A `keras.Model` instance.
"""

EfficientNetV2.__doc__ = __v2_head_doc__ + """
Args:
  model_type: is the pre-defined model, value in ["t", "s", "m", "l", "xl", "b0", "b1", "b2", "b3"].
  model_name: string, model name.
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]) + """
Model architectures:
  | V2 Model                   | Params | FLOPs  | Input | Top1 Acc |
  | -------------------------- | ------ | ------ | ----- | -------- |
  | EfficientNetV2B0           | 7.1M   | 0.72G  | 224   | 78.7     |
  | - ImageNet21k-ft1k         | 7.1M   | 0.72G  | 224   | 77.55?   |
  | EfficientNetV2B1           | 8.1M   | 1.21G  | 240   | 79.8     |
  | - ImageNet21k-ft1k         | 8.1M   | 1.21G  | 240   | 79.03?   |
  | EfficientNetV2B2           | 10.1M  | 1.71G  | 260   | 80.5     |
  | - ImageNet21k-ft1k         | 10.1M  | 1.71G  | 260   | 79.48?   |
  | EfficientNetV2B3           | 14.4M  | 3.03G  | 300   | 82.1     |
  | - ImageNet21k-ft1k         | 14.4M  | 3.03G  | 300   | 82.46?   |
  | EfficientNetV2T            | 13.6M  | 3.18G  | 288   | 82.34    |
  | EfficientNetV2T_GC         | 13.7M  | 3.19G  | 288   | 82.46    |
  | EfficientNetV2S            | 21.5M  | 8.41G  | 384   | 83.9     |
  | - ImageNet21k-ft1k         | 21.5M  | 8.41G  | 384   | 84.9     |
  | EfficientNetV2M            | 54.1M  | 24.69G | 480   | 85.2     |
  | - ImageNet21k-ft1k         | 54.1M  | 24.69G | 480   | 86.2     |
  | EfficientNetV2L            | 119.5M | 56.27G | 480   | 85.7     |
  | - ImageNet21k-ft1k         | 119.5M | 56.27G | 480   | 86.9     |
  | EfficientNetV2XL, 21k-ft1k | 206.8M | 93.66G | 512   | 87.2     |

Training configures: `Eval size` is used as the default model `input_shape` for each model type.
  | Model            | Train size | Eval size | Dropout | Randaug | Mixup |
  | ---------------- | ---------- | --------- | ------- | ------- | ----- |
  | EfficientNetV2B0 | 192        | 224       | 0.2     | 0       | 0     |
  | EfficientNetV2B1 | 192        | 240       | 0.2     | 0       | 0     |
  | EfficientNetV2B2 | 208        | 260       | 0.3     | 0       | 0     |
  | EfficientNetV2B3 | 240        | 300       | 0.3     | 0       | 0     |
  | EfficientNetV2T  | 224        | 320       | 0.2     | 0       | 0     |
  | EfficientNetV2S  | 300        | 384       | 0.2     | 10      | 0     |
  | EfficientNetV2M  | 384        | 480       | 0.3     | 15      | 0.2   |
  | EfficientNetV2L  | 384        | 480       | 0.4     | 20      | 0.5   |
  | EfficientNetV2XL | 384        | 512       | 0.4     | 20      | 0.5   |
"""

__v2_default_doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]) + """
Training configures: `Eval size` is used as the default model `input_shape`.
  | Model   | Train size | Eval size | Dropout | Randaug | Mixup |
  | ------- | ---------- | --------- | ------- | ------- | ----- |
  {train_config}
"""

EfficientNetV2B0.__doc__ = __v2_default_doc__.format(train_config="| EffV2B0 | 192        | 224       | 0.2     | 0       | 0     |")
EfficientNetV2B1.__doc__ = __v2_default_doc__.format(train_config="| EffV2B1 | 192        | 240       | 0.2     | 0       | 0     |")
EfficientNetV2B2.__doc__ = __v2_default_doc__.format(train_config="| EffV2B2 | 208        | 260       | 0.3     | 0       | 0     |")
EfficientNetV2B3.__doc__ = __v2_default_doc__.format(train_config="| EffV2B3 | 240        | 300       | 0.3     | 0       | 0     |")
EfficientNetV2S.__doc__ = __v2_default_doc__.format(train_config="| EffV2S  | 300        | 384       | 0.2     | 10      | 0     |")
EfficientNetV2M.__doc__ = __v2_default_doc__.format(train_config="| EffV2M  | 384        | 480       | 0.3     | 15      | 0.2   |")
EfficientNetV2L.__doc__ = __v2_default_doc__.format(train_config="| EffV2L  | 384        | 480       | 0.4     | 20      | 0.5   |")
EfficientNetV2XL.__doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet21k", "imagenet21k-ft1k"]) + """
Training configures: `Eval size` is used as the default model `input_shape`.
  | Model   | Train size | Eval size | Dropout | Randaug | Mixup |
  | ------- | ---------- | --------- | ------- | ------- | ----- |
  | EffV2XL | 384        | 512       | 0.4     | 20      | 0.5   |
"""
EfficientNetV2T.__doc__ = __v2_head_doc__ + """Architecture and weights from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models#july-5-9-2021).

Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet"])
EfficientNetV2T_GC.__doc__ = EfficientNetV2T.__doc__

__v1_head_doc__ = """
Keras implementation of [Github tensorflow/tpu/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
Paper [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf).
"""

EfficientNetV1.__doc__ = __v1_head_doc__ + """
Args:
  model_type: is the pre-defined model, value in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "l2"].
  model_name: string, model name.
""" + __tail_doc__.format(pretrained=[None, "imagenet", "noisy_student"]) + """
Model architectures:
  | V1 Model                       | Params | FLOPs   | Input | Top1 Acc |
  | ------------------------------ | ------ | ------- | ----- | -------- |
  | EfficientNetV1B0               | 5.3M   | 0.39G   | 224   | 77.6     |
  | - NoisyStudent                 | 5.3M   | 0.39G   | 224   | 78.8     |
  | EfficientNetV1B1               | 7.8M   | 0.70G   | 240   | 79.6     |
  | - NoisyStudent                 | 7.8M   | 0.70G   | 240   | 81.5     |
  | EfficientNetV1B2               | 9.1M   | 1.01G   | 260   | 80.5     |
  | - NoisyStudent                 | 9.1M   | 1.01G   | 260   | 82.4     |
  | EfficientNetV1B3               | 12.2M  | 1.86G   | 300   | 81.9     |
  | - NoisyStudent                 | 12.2M  | 1.86G   | 300   | 84.1     |
  | EfficientNetV1B4               | 19.3M  | 4.46G   | 380   | 83.3     |
  | - NoisyStudent                 | 19.3M  | 4.46G   | 380   | 85.3     |
  | EfficientNetV1B5               | 30.4M  | 10.40G  | 456   | 84.3     |
  | - NoisyStudent                 | 30.4M  | 10.40G  | 456   | 86.1     |
  | EfficientNetV1B6               | 43.0M  | 19.29G  | 528   | 84.8     |
  | - NoisyStudent                 | 43.0M  | 19.29G  | 528   | 86.4     |
  | EfficientNetV1B7               | 66.3M  | 38.13G  | 600   | 85.2     |
  | - NoisyStudent                 | 66.3M  | 38.13G  | 600   | 86.9     |
  | EfficientNetV1L2, NoisyStudent | 480.3M | 477.98G | 800   | 88.4     |


Training configures:
  | Model            | Input resolution | Dropout | Drop connect rate |
  | ---------------- | ---------------- | ------- | ----------------- |
  | EfficientNetV1B0 | 224              | 0.2     | 0.2               |
  | EfficientNetV1B1 | 240              | 0.2     | 0.2               |
  | EfficientNetV1B2 | 260              | 0.3     | 0.2               |
  | EfficientNetV1B3 | 300              | 0.3     | 0.2               |
  | EfficientNetV1B4 | 380              | 0.4     | 0.2               |
  | EfficientNetV1B5 | 456              | 0.4     | 0.2               |
  | EfficientNetV1B6 | 528              | 0.5     | 0.2               |
  | EfficientNetV1B7 | 600              | 0.5     | 0.2               |
  | EfficientNetV1L2 | 800              | 0.5     | 0.2               |
"""

__v1_default_doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None, "imagenet", "noisy_student"]) + """
Training configures: `Eval size` is used as the default model `input_shape`.
  | Model            | Input resolution | Dropout | Drop connect rate |
  | ---------------- | ---------------- | ------- | ----------------- |
  {train_config}
"""

EfficientNetV1B0.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B0 | 224              | 0.2     | 0.2               |")
EfficientNetV1B1.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B1 | 240              | 0.2     | 0.2               |")
EfficientNetV1B2.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B2 | 260              | 0.3     | 0.2               |")
EfficientNetV1B3.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B3 | 300              | 0.3     | 0.2               |")
EfficientNetV1B4.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B4 | 380              | 0.4     | 0.2               |")
EfficientNetV1B5.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B5 | 456              | 0.4     | 0.2               |")
EfficientNetV1B6.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B6 | 528              | 0.5     | 0.2               |")
EfficientNetV1B7.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1B7 | 600              | 0.5     | 0.2               |")
EfficientNetV1L2.__doc__ = __v1_default_doc__.format(train_config="| EfficientNetV1L2 | 800              | 0.5     | 0.2               |")

__v1_lite_default_doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained=[None]) + """
Training configures: `Eval size` is used as the default model `input_shape`.
  | Model               | Input resolution | Dropout | Drop connect rate |
  | ------------------- | ---------------- | ------- | ----------------- |
  {train_config}
"""
EfficientNetV1Lite0.__doc__ = __v1_lite_default_doc__.format(train_config="| EfficientNetV1Lite0 | 320              | 0.2     | 0.2               |")
EfficientNetV1Lite1.__doc__ = __v1_lite_default_doc__.format(train_config="| EfficientNetV1Lite1 | 384              | 0.2     | 0.2               |")
EfficientNetV1Lite2.__doc__ = __v1_lite_default_doc__.format(train_config="| EfficientNetV1Lite2 | 448              | 0.3     | 0.2               |")
EfficientNetV1Lite3.__doc__ = __v1_lite_default_doc__.format(train_config="| EfficientNetV1Lite3 | 512              | 0.3     | 0.2               |")
EfficientNetV1Lite4.__doc__ = __v1_lite_default_doc__.format(train_config="| EfficientNetV1Lite4 | 640              | 0.4     | 0.2               |")
