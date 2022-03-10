from keras_cv_attention_models.coatnet.coatnet import (
    CoAtNet,
    CoAtNetT,
    CoAtNet0,
    CoAtNet1,
    CoAtNet2,
    CoAtNet3,
    CoAtNet4,
    CoAtNet5,
    CoAtNet6,
    CoAtNet7,
    mhsa_with_multi_head_relative_position_embedding,
)

__head_doc__ = """
Keras implementation of `CoAtNet`, or `Conv + Transformer` networks.
Paper [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf).
"""

__tail_doc__ = """  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `res_MBConv` block.
      - `transfrom` or any not `c` / `C` starts word, means `res_mhsa` + `res_ffn` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  expansion: filter expansion in each block. The larger the wider.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack.
      Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  head_dimension: heads number for transformer block. `128` for `CoAtNet6` and `CoAtNet7`, `64` for `CoAtNet5`, `32` for others.
  use_dw_strides: boolean value if using strides on `Conv2D` or next `DepthWiseConv2D` for `res_MBConv` block.
      It's claimed higher accuracy for small models using strides on `DepthWiseConv2D`. Default True.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: None or "imagenet". Currently only `CoAtNet0` with "imagenet" available.
  **kwargs: other parameters if available.
Returns:
    A `keras.Model` instance.
"""

CoAtNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                               | Params | Image resolution | Top1 Acc |
  | ----------------------------------- | ------ | ---------------- | -------- |
  | CoAtNet0                            | 25M    | 224              | 81.6     |
  | CoAtNet0, Strided DConv             | 25M    | 224              | 82.0     |
  | CoAtNet0                            | 25M    | 384              | 83.9     |
  | CoAtNet1                            | 42M    | 224              | 83.3     |
  | CoAtNet1, Strided DConv             | 42M    | 224              | 83.5     |
  | CoAtNet1                            | 42M    | 384              | 85.1     |
  | CoAtNet2                            | 75M    | 224              | 84.1     |
  | CoAtNet2, Strided DConv             | 75M    | 224              | 84.1     |
  | CoAtNet2                            | 75M    | 384              | 85.7     |
  | CoAtNet2                            | 75M    | 512              | 85.9     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 384              | 87.1     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 512              | 87.3     |
  | CoAtNet3                            | 168M   | 224              | 84.5     |
  | CoAtNet3                            | 168M   | 384              | 85.8     |
  | CoAtNet3                            | 168M   | 512              | 86.0     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 384              | 87.6     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 384              | 87.9     |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 384              | 88.4     |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |

  **JFT pre-trained models accuracy**

  | Model    | Image resolution | Reported Params | self-defined Params | Top1 Acc |
  | -------- | ---------------- | --------------- | ------------------- | -------- |
  | CoAtNet3 | 384              | 168M            | 162.96M             | 88.52    |
  | CoAtNet3 | 512              | 168M            | 163.57M             | 88.81    |
  | CoAtNet4 | 512              | 275M            | 273.10M             | 89.11    |
  | CoAtNet5 | 512              | 688M            | 680.47M             | 89.77    |
  | CoAtNet6 | 512              | 1.47B           | 1.340B              | 90.45    |
  | CoAtNet7 | 512              | 2.44B           | 2.422B              | 90.88    |
"""

CoAtNet0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CoAtNet1.__doc__ = CoAtNet0.__doc__
CoAtNet2.__doc__ = CoAtNet0.__doc__
CoAtNet3.__doc__ = CoAtNet0.__doc__
CoAtNet4.__doc__ = CoAtNet0.__doc__
CoAtNet5.__doc__ = CoAtNet0.__doc__
CoAtNet6.__doc__ = CoAtNet0.__doc__
CoAtNet7.__doc__ = CoAtNet0.__doc__

mhsa_with_multi_head_relative_position_embedding.__doc__ = __head_doc__ + """
Multi head self attention with multi head relative positional embedding. Defined as function, not layer.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.mhsa_with_multi_head_relative_position_embedding(inputs, num_heads=4, out_shape=512)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 512])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'conv2d_2/kernel:0': TensorShape([1, 1, 256, 1024]),
#  'multi_head_relative_positional_embedding_1/pos_emb:0': TensorShape([837, 4]),
#  'dense_2/kernel:0': TensorShape([512, 512])}
"""
