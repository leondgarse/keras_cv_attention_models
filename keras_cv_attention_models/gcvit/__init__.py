from keras_cv_attention_models.gcvit.gcvit import GCViT, GCViT_XXTiny, GCViT_XTiny, GCViT_Tiny, GCViT_Tiny2, GCViT_Small, GCViT_Small2, GCViT_Base, GCViT_Large

__head_doc__ = """
Keras implementation of [Github NVlabs/GCVit](https://github.com/NVlabs/GCVit).
Paper [PDF 2206.09959 Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf).
"""

__tail_doc__ = """  window_ratios: window split ratio. Each stack will calculate `window_size = (height // window_ratio, width // window_ratio)` .
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
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
  pretrained: None or one of ["imagenet", "imagenet21k-ft1k"].
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

GCViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  num_heads: num heads for each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model           | Params | FLOPs  | Input | Top1 Acc |
  | --------------- | ------ | ------ | ----- | -------- |
  | GCViT_XXTiny    | 12.0M  | 2.15G  | 224   | 79.9     |
  | GCViT_XTiny     | 20.0M  | 2.96G  | 224   | 82.0     |
  | GCViT_Tiny      | 28.2M  | 4.83G  | 224   | 83.5     |
  | GCViT_Tiny2     | 34.5M  | 6.28G  | 224   | 83.7     |
  | GCViT_Small     | 51.1M  | 8.63G  | 224   | 84.3     |
  | GCViT_Small2    | 68.6M  | 11.7G  | 224   | 84.8     |
  | GCViT_Base      | 90.3M  | 14.9G  | 224   | 85.0     |
  | GCViT_Large     | 202.1M | 32.8G  | 224   | 85.7     |
  | - 21k_ft1k      | 202.1M | 32.8G  | 224   | 86.6     |
  | - 21k_ft1k, 384 | 202.9M | 105.1G | 384   | 87.4     |
  | - 21k_ft1k, 512 | 203.8M | 205.1G | 512   | 87.6     |
"""

GCViT_XXTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

GCViT_XTiny.__doc__ = GCViT_XXTiny.__doc__
GCViT_Tiny.__doc__ = GCViT_XXTiny.__doc__
GCViT_Tiny2.__doc__ = GCViT_XXTiny.__doc__
GCViT_Small.__doc__ = GCViT_XXTiny.__doc__
GCViT_Small2.__doc__ = GCViT_XXTiny.__doc__
GCViT_Base.__doc__ = GCViT_XXTiny.__doc__
GCViT_Large.__doc__ = GCViT_XXTiny.__doc__
