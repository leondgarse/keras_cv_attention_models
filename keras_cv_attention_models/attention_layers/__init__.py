
from keras_cv_attention_models.common_layers import (
    activation_by_name,
    anti_alias_downsample,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    deep_stem,
    fold_by_conv2d_transpose,
    quad_stem,
    tiered_stem,
    drop_block,
    drop_connect_rates_split,
    eca_module,
    hard_swish,
    layer_norm,
    make_divisible,
    output_block,
    se_module,
    tpu_compatible_extract_patches,
    unfold_by_conv2d,
)
from keras_cv_attention_models.aotnet.aotnet import aot_stack, aot_block
from keras_cv_attention_models.botnet.botnet import RelativePositionalEmbedding, mhsa_with_relative_position_embedding
from keras_cv_attention_models.cotnet.cotnet import cot_attention
from keras_cv_attention_models.coat.coat import ConvPositionalEncoding, ConvRelativePositionalEncoding
from keras_cv_attention_models.halonet.halonet import halo_attention
from keras_cv_attention_models.resnest.resnest import rsoftmax, split_attention_conv2d
from keras_cv_attention_models.volo.volo import outlook_attention, outlook_attention_simple, BiasLayer, PositionalEmbedding, ClassToken
from keras_cv_attention_models.mlp_family.mlp_mixer import mlp_block, mixer_block
from keras_cv_attention_models.mlp_family.res_mlp import ChannelAffine
from keras_cv_attention_models.mlp_family.gated_mlp import spatial_gating_block
from keras_cv_attention_models.levit.levit import MultiHeadPositionalEmbedding, mhsa_with_multi_head_position_and_strides
from keras_cv_attention_models.nfnets.nfnets import ScaledStandardizedConv2D, ZeroInitGain
from keras_cv_attention_models.beit.beit import MultiHeadRelativePositionalEmbedding

unfold_by_conv2d.__doc__ = """
For issue https://github.com/leondgarse/keras_cv_attention_models/issues/8,
`tf.image.extract_patches` NOT working for TPU.

input: `[batch, height, width, channel]`.
output (compressed=True): `[batch, height // strides,  width // strides, height_kernel * width_kernel * channel]`.
output (compressed=False): `[batch, height // strides,  width // strides, height_kernel, width_kernel, channel]`.

Args:
  inputs: input tensor.
  kernel_size: same as `Conv2D` kernel_size. Default `3`.
  strides: same as `Conv2D` strides. Default `2`.
  dilation_rate: same as `Conv2D` dilation_rate. Default `1`.
  padding: "VALID" or "SAME", will perform padding in PyTorch way if "SAME". Default "SAME".
  compressed: boolean value if compress extracted `height_kernel`, `width_kernel`, `channel` into 1 dimension. Default `True`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = tf.ones([1, 64, 27, 192])
>>> print(attention_layers.unfold_by_conv2d(aa, kernel_size=3, strides=2).shape)
# (1, 32, 14, 1728)
>>> print(attention_layers.unfold_by_conv2d(aa, kernel_size=3, strides=2, compressed=False).shape)
# (1, 32, 14, 3, 3, 192)

# Performs slower than `extract_patches`
>>> inputs = keras.layers.Input([64, 27, 192])
>>> mm = keras.models.Model(inputs, attention_layers.unfold_by_conv2d(inputs))
>>> %timeit mm(aa)
# 1.3 ms ± 4.93 µs per loop
>>> mm = keras.models.Model(inputs, tf.image.extract_patches(inputs, [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME'))
>>> %timeit mm(aa)
# 208 µs ± 630 ns per loop
"""

fold_by_conv2d_transpose.__doc__ = """
Fold function like `torch.nn.Fold` using `Conv2DTranspose`.

input (compressed=True): `[batch, height // strides,  width // strides, height_kernel * width_kernel * channel]`.
input (compressed=False): `[batch, height // strides,  width // strides, height_kernel, width_kernel, channel]`.
output: `[batch, height, width, channel]`.

Args:
  patches: input tensor.
  output_shape: specific output shape in format `(height, width)`.
      Default `None` will just cut `padded` by `out[:, paded:-paded, paded:-paded, :]`.
  kernel_size: same as `Conv2DTranspose` kernel_size. Default `3`.
  strides: same as `Conv2DTranspose` strides. Default `2`.
  dilation_rate: same as `Conv2DTranspose` dilation_rate. Default `1`.
  padding: "VALID" or "SAME", indicates if `patches` is generated from a padded input. Default "SAME".
  compressed: boolean value if `patches` last dimension is a compressed of `height_kernel`, `width_kernel`, `channel`.
      Default "auto" means auto judge from `patches` shape length.

Examples:

>>> inputs = np.random.uniform(size=[1, 64, 27, 192]).astype("float32")
>>> kernel_size, strides = 3, 2

>>> # ==== Torch unfold - fold ====
>>> import torch
>>> torch_inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)  # NCHW
>>> unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
>>> fold = torch.nn.Fold(output_size=torch_inputs.shape[2:4], kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
>>> torch_patches = unfold(torch_inputs)
>>> torch_fold = fold(torch_patches)

>>> # ==== TF unfold - fold ====
>>> from keras_cv_attention_models import attention_layers
>>> tf_patches = attention_layers.unfold_by_conv2d(inputs, kernel_size, strides)
>>> tf_fold = attention_layers.fold_by_conv2d_transpose(tf_patches, output_shape=inputs.shape[1:], kernel_size=kernel_size, strides=strides)
>>> print(f"{np.allclose(tf_fold, torch_fold.permute([0, 2, 3, 1])) = }")
# np.allclose(tf_fold, torch_fold.permute([0, 2, 3, 1])) = True

>>> # ==== TF extract_patches ====
>>> pad = kernel_size // 2
>>> pad_inputs = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
>>> patches = tf.image.extract_patches(pad_inputs, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
>>> print(f"{np.allclose(tf_patches, patches) = }")
# np.allclose(tf_patches, patches) = True
"""
