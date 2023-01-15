from keras_cv_attention_models.gpvit.gpvit import GPViT, GPViT_L1, GPViT_L2, GPViT_L3, GPViT_L4, PureWeigths

__head_doc__ = """
Keras implementation of [Github ChenhongyiYang/GPViT](https://github.com/ChenhongyiYang/GPViT).
Paper [PDF 2212.06795 GPVIT: A HIGH RESOLUTION NON-HIERARCHICAL VISION TRANSFORMER WITH GROUP PROPAGATION](https://arxiv.org/pdf/2212.06795.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scale: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
      Default 0 for not using.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

GPViT.__doc__ = __head_doc__ + """
Args:
  num_layers: number of transformer blocks.
  embed_dims: output channels for each stack.
  stem_depth: number of stem conv blocks.
  num_window_heads: number of heads for `window_lepe_attention_mlp_block` blocks.
  num_group_heads: number of heads for `group_attention` blocks.
  mlp_ratios: int value indicates expand ratio for mlp blocks hidden channel in each stack.
  window_size: number of `window_size` for `window_lepe_attention_mlp_block` blocks.
  group_attention_layer_ids: list of layer id for using `group_attention`, others will be `window_lepe_attention_mlp_block`.
  group_attention_layer_group_tokens: list of `num_group_token` for each block using `group_attention`.
  use_neck_attention_output: boolean value whether using `light_group_attention` before output block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model    | Params | FLOPs  | Input | Top1 Acc |
  | -------- | ------ | ------ | ----- | -------- |
  | GPViT_L1 | 9.59M  | 6.15G  | 224   | 80.5     |
  | GPViT_L2 | 24.2M  | 15.74G | 224   | 83.4     |
  | GPViT_L3 | 36.7M  | 23.54G | 224   | 84.1     |
  | GPViT_L4 | 75.5M  | 48.29G | 224   | 84.3     |
"""

GPViT_L1.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

GPViT_L2.__doc__ = GPViT_L1.__doc__
GPViT_L3.__doc__ = GPViT_L1.__doc__
GPViT_L4.__doc__ = GPViT_L1.__doc__
