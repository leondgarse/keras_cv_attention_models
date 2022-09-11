from keras_cv_attention_models.hornet.hornet import (
    HorNet,
    HorNetTiny,
    HorNetTinyGF,
    HorNetSmall,
    HorNetSmallGF,
    HorNetBase,
    HorNetBaseGF,
    HorNetLarge,
    HorNetLargeGF,
    ComplexDense,
    global_local_filter,
    gnconv,
)

__head_doc__ = """
Keras implementation of [Github raoyongming/hornet](https://github.com/raoyongming/hornet).
Paper [PDF 2207.14284 HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/pdf/2207.14284.pdf).
"""

__tail_doc__ = """  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: one of None or "imagenet".
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

HorNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  gn_split: int or list of int value. Number of channels to split in `gnconv` block for each stack. Default `[2, 3, 4, 5]`
  use_global_local_filter: boolean or list of boolean value. Whether use `global_local_filter` or not for each stack.
      Default `[False, False, True, True]` for `HorNet*GF` models, `False` for others.
  scale: float or list of float value. Multiply scale for depth wise output in `gnconv` block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | FLOPs  | Input | Top1 Acc |
  | ------------- | ------ | ------ | ----- | -------- |
  | HorNetTiny    | 22.4M  | 4.01G  | 224   | 82.8     |
  | HorNetTinyGF  | 23.0M  | 3.94G  | 224   | 83.0     |
  | HorNetSmall   | 49.5M  | 8.87G  | 224   | 83.8     |
  | HorNetSmallGF | 50.4M  | 8.77G  | 224   | 84.0     |
  | HorNetBase    | 87.3M  | 15.65G | 224   | 84.2     |
  | HorNetBaseGF  | 88.4M  | 15.51G | 224   | 84.3     |
  | HorNetLarge   | 194.5M | 34.91G | 224   | 86.8     |
  | HorNetLargeGF | 196.3M | 34.72G | 224   | 87.0     |
  | HorNetLargeGF | 201.8M | 102.0G | 384   | 87.7     |
"""

HorNetTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

HorNetTinyGF.__doc__ = HorNetTiny.__doc__
HorNetSmall.__doc__ = HorNetTiny.__doc__
HorNetSmallGF.__doc__ = HorNetTiny.__doc__
HorNetBase.__doc__ = HorNetTiny.__doc__
HorNetBaseGF.__doc__ = HorNetTiny.__doc__
HorNetLarge.__doc__ = HorNetTiny.__doc__
HorNetLargeGF.__doc__ = HorNetTiny.__doc__

ComplexDense.__doc__ = __head_doc__
global_local_filter.__doc__ = __head_doc__
gnconv.__doc__ = __head_doc__
