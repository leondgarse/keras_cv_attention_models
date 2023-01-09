from keras_cv_attention_models.convnext.convnext import (
    global_response_normalize,
    ConvNeXt,
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtXlarge,
)
from keras_cv_attention_models.convnext.convnext_v2 import (
    ConvNeXtV2,
    ConvNeXtV2Atto,
    ConvNeXtV2Femto,
    ConvNeXtV2Pico,
    ConvNeXtV2Nano,
    ConvNeXtV2Tiny,
    ConvNeXtV2Base,
    ConvNeXtV2Large,
    ConvNeXtV2Huge,
)

__v1_head_doc__ = """
Keras implementation of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
Paper [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf).
"""

__common_head_doc__ = """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block. Default `-1` means using `out_channels[0]`.
  model_name: string, model name.
"""

__common_tail_doc__ = """  layer_scale_init_value: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
      Default 1e-6 for v1, 0 for v2.
  use_grn: boolean value if use `global_response_normalize` block. Default False for v1, True for v2.
  head_init_scale: init head layer scale value if create model from scratch or fine-tune. Default `1`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch. or `0` to disable.
      Default 0.1.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  dropout: dropout rate if top layers is included.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      or 'imagenet21k-ft1k' (pre-trined on ImageNet21k, fine-tuning on ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

__v1_tail_doc__ = """
Model architectures:
  | Model               | Params | FLOPs   | Input | Top1 Acc |
  | ------------------- | ------ | ------- | ----- | -------- |
  | ConvNeXtTiny        | 28M    | 4.49G   | 224   | 82.1     |
  | - ImageNet21k-ft1k  | 28M    | 4.49G   | 224   | 82.9     |
  | - ImageNet21k-ft1k  | 28M    | 13.19G  | 384   | 84.1     |
  | ConvNeXtSmall       | 50M    | 8.73G   | 224   | 83.1     |
  | - ImageNet21k-ft1k  | 50M    | 8.73G   | 224   | 84.6     |
  | - ImageNet21k-ft1k  | 50M    | 25.67G  | 384   | 85.8     |
  | ConvNeXtBase        | 89M    | 15.42G  | 224   | 83.8     |
  | ConvNeXtBase        | 89M    | 45.32G  | 384   | 85.1     |
  | - ImageNet21k-ft1k  | 89M    | 15.42G  | 224   | 85.8     |
  | - ImageNet21k-ft1k  | 89M    | 45.32G  | 384   | 86.8     |
  | ConvNeXtLarge       | 198M   | 34.46G  | 224   | 84.3     |
  | ConvNeXtLarge       | 198M   | 101.28G | 384   | 85.5     |
  | - ImageNet21k-ft1k  | 198M   | 34.46G  | 224   | 86.6     |
  | - ImageNet21k-ft1k  | 198M   | 101.28G | 384   | 87.5     |
  | ConvNeXtXLarge, 21k | 350M   | 61.06G  | 224   | 87.0     |
  | ConvNeXtXLarge, 21k | 350M   | 179.43G | 384   | 87.8     |
"""

ConvNeXt.__doc__ = __v1_head_doc__ + __common_head_doc__ + __common_tail_doc__ + __v1_tail_doc__

ConvNeXtTiny.__doc__ = __v1_head_doc__ + """
Args:
""" + __common_tail_doc__ + __v1_tail_doc__

ConvNeXtSmall.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtBase.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtLarge.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtLarge.__doc__ = ConvNeXtTiny.__doc__

__v2_head_doc__ = """
Keras implementation of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt-V2).
Paper [PDF 2301.00808 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808.pdf).
"""

__v2_tail_doc__ = """
Model architectures:
  | Model              | Params | FLOPs  | Input | Top1 Acc |
  | ------------------ | ------ | ------ | ----- | -------- |
  | ConvNeXtV2Atto     | 3.7M   | 0.55G  | 224   | 76.7     |
  | ConvNeXtV2Femto    | 5.2M   | 0.78G  | 224   | 78.5     |
  | ConvNeXtV2Pico     | 9.1M   | 1.37G  | 224   | 80.3     |
  | ConvNeXtV2Nano     | 15.6M  | 2.45G  | 224   | 81.9     |
  | - ImageNet21k-ft1k | 15.6M  | 2.45G  | 224   | 82.1     |
  | - ImageNet21k-ft1k | 15.6M  | 7.21G  | 384   | 83.4     |
  | ConvNeXtV2Tiny     | 28.6M  | 4.47G  | 224   | 83.0     |
  | - ImageNet21k-ft1k | 28.6M  | 4.47G  | 224   | 83.9     |
  | - ImageNet21k-ft1k | 28.6M  | 13.1G  | 384   | 85.1     |
  | ConvNeXtV2Base     | 89M    | 15.4G  | 224   | 84.9     |
  | - ImageNet21k-ft1k | 89M    | 15.4G  | 224   | 86.8     |
  | - ImageNet21k-ft1k | 89M    | 45.2G  | 384   | 87.7     |
  | ConvNeXtV2Large    | 198M   | 34.4G  | 224   | 85.8     |
  | - ImageNet21k-ft1k | 198M   | 34.4G  | 224   | 87.3     |
  | - ImageNet21k-ft1k | 198M   | 101.1G | 384   | 88.2     |
  | ConvNeXtV2Huge     | 660M   | 115G   | 224   | 86.3     |
  | - ImageNet21k-ft1k | 660M   | 337.9G | 384   | 88.7     |
  | - ImageNet21k-ft1k | 660M   | 600.8G | 512   | 88.9     |
"""

ConvNeXtV2.__doc__ = __v2_head_doc__ + __common_head_doc__ + __common_tail_doc__ + __v2_tail_doc__

ConvNeXtV2Atto.__doc__ = __v2_head_doc__ + """
Args:
""" + __common_tail_doc__ + __v2_tail_doc__

ConvNeXtV2Femto.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Pico.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Nano.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Tiny.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Base.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Large.__doc__ = ConvNeXtV2Atto.__doc__
ConvNeXtV2Huge.__doc__ = ConvNeXtV2Atto.__doc__

global_response_normalize.__doc__ = __v2_head_doc__ + """
Global Response Normalize block

Args:
  inputs: input tensor.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.global_response_normalize(inputs)
>>> print(f"{nn.shape = }")
nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
{'channel_affine/weight:0': TensorShape([256]), 'channel_affine/bias:0': TensorShape([256])}
"""
