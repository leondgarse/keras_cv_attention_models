from keras_cv_attention_models.fastvit.fastvit import (
    FastViT,
    FastViT_T8,
    FastViT_T12,
    FastViT_S12,
    FastViT_SA12,
    FastViT_SA24,
    FastViT_SA36,
    FastViT_MA36,
    # switch_to_deploy,
)

__head_doc__ = """
Keras implementation of [Github NVlabs/FasterViT](https://github.com/NVlabs/FasterViT).
Paper [PDF 2306.06189 FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/pdf/2306.06189.pdf).
"""

__tail_doc__ = """  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `rep_conv_block` block.
      - `transfrom` or any not `c` / `C` starts word, means `multi_head_self_attention` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
      `["conv", "conv", "conv", "conv"]` for SA models, all conv for others.
  layer_scale: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
      Default 1e-6 for SA models, 1e-5 for others.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  deploy: boolean value if build a fused model. **Evaluation only, not good for training**.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `hard_swish`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: top dropout rate if top layers is included. Default 0.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

FastViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of block for each stack.
  out_channels: output channels for each stack.
  stem_width: channel dimension output for stem block, default -1 for using out_channels[0].
  mlp_ratio: int value for mlp_ratio for each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs | Input | Top1 Acc |
  | ------------ | ------ | ----- | ----- | -------- |
  | FastViT_T8   | 4.03M  | 0.65G | 256   | 76.2     |
  | - distill    | 4.03M  | 0.65G | 256   | 77.2     |
  | FastViT_T12  | 7.55M  | 1.34G | 256   | 79.3     |
  | - distill    | 7.55M  | 1.34G | 256   | 80.3     |
  | FastViT_S12  | 9.47M  | 1.74G | 256   | 79.9     |
  | - distill    | 9.47M  | 1.74G | 256   | 81.1     |
  | FastViT_SA12 | 11.58M | 1.88G | 256   | 80.9     |
  | - distill    | 11.58M | 1.88G | 256   | 81.9     |
  | FastViT_SA24 | 21.55M | 3.66G | 256   | 82.7     |
  | - distill    | 21.55M | 3.66G | 256   | 83.4     |
  | FastViT_SA36 | 31.53M | 5.44G | 256   | 83.6     |
  | - distill    | 31.53M | 5.44G | 256   | 84.2     |
  | FastViT_MA36 | 44.07M | 7.64G | 256   | 83.9     |
  | - distill    | 44.07M | 7.64G | 256   | 84.6     |
"""

FastViT_T8.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

FastViT_T12.__doc__ = FastViT_T8.__doc__
FastViT_S12.__doc__ = FastViT_T8.__doc__
FastViT_SA12.__doc__ = FastViT_T8.__doc__
FastViT_SA24.__doc__ = FastViT_T8.__doc__
FastViT_SA36.__doc__ = FastViT_T8.__doc__
FastViT_MA36.__doc__ = FastViT_T8.__doc__
