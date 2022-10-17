from keras_cv_attention_models.maxvit.maxvit import MaxViT, MaxViT_T, MaxViT_S, MaxViT_B, MaxViT_L, MaxViT_XL

__head_doc__ = """
Keras implementation of [Github google-research/maxvit](https://github.com/google-research/maxvit).
Paper [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).
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

MaxViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | FLOPs  | Input | Top1 Acc |
  | ------------- | ------ | ------ | ----- | -------- |
  | MaxViT_T      | 31M    | 5.6G   | 224   | 83.62    |
"""

MaxViT_T.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

MaxViT_S.__doc__ = MaxViT_T.__doc__
MaxViT_B.__doc__ = MaxViT_T.__doc__
MaxViT_L.__doc__ = MaxViT_T.__doc__
MaxViT_XL.__doc__ = MaxViT_T.__doc__
