from keras_cv_attention_models.hiera.hiera import Hiera, HieraTiny, HieraSmall, HieraBase, HieraBasePlus, HieraLarge, HieraHuge

__head_doc__ = """
Keras implementation of [Github facebookresearch/hiera](https://github.com/facebookresearch/hiera).
Paper [PDF 2306.00989 Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/pdf/2306.00989.pdf).
"""

__tail_doc__ = """  strides: list of int indicates strides for each stack. Default `[1, 2, 2, 2]`.
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
  pretrained: one of None or "mae_in1k_ft1k".
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

Hiera.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  num_heads: int or list value for num heads in each stack.
  use_window_attentions: boolean or list value, each value in the list can also be a list of boolean.
      Indicates if use window attention in each stack.
      Element value like `[True, False]` means first one is True, others are False.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params  | FLOPs   | Input | Top1 Acc |
  | ------------- | ------- | ------- | ----- | -------- |
  | HieraTiny     | 27.91M  | 4.93G   | 224   | 82.8     |
  | HieraSmall    | 35.01M  | 6.44G   | 224   | 83.8     |
  | HieraBase     | 51.52M  | 9.43G   | 224   | 84.5     |
  | HieraBasePlus | 69.90M  | 12.71G  | 224   | 85.2     |
  | HieraLarge    | 213.74M | 40.43G  | 224   | 86.1     |
  | HieraHuge     | 672.78M | 125.03G | 224   | 86.9     |
"""

HieraTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

HieraSmall.__doc__ = HieraTiny.__doc__
HieraBase.__doc__ = HieraTiny.__doc__
HieraBasePlus.__doc__ = HieraTiny.__doc__
HieraLarge.__doc__ = HieraTiny.__doc__
HieraHuge.__doc__ = HieraTiny.__doc__
