
from keras_cv_attention_models.common_layers import (
    activation_by_name,
    anti_alias_downsample,
    batchnorm_with_activation,
    conv2d_no_bias,
    CompatibleExtractPatches,
    depthwise_conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    EvoNormalization,
    eca_module,
    fold_by_conv2d_transpose,
    group_norm,
    hard_swish,
    layer_norm,
    make_divisible,
    output_block,
    se_module,
    PreprocessInput,
    imagenet_decode_predictions,
    add_pre_post_process,
)
from keras_cv_attention_models.aotnet.aotnet import aot_stack, aot_block, deep_stem, quad_stem, tiered_stem
from keras_cv_attention_models.botnet.botnet import RelativePositionalEmbedding, mhsa_with_relative_position_embedding
from keras_cv_attention_models.cotnet.cotnet import cot_attention
from keras_cv_attention_models.coat.coat import ConvPositionalEncoding, ConvRelativePositionalEncoding
from keras_cv_attention_models.halonet.halonet import halo_attention
from keras_cv_attention_models.resnest.resnest import rsoftmax, split_attention_conv2d
from keras_cv_attention_models.volo.volo import outlook_attention, outlook_attention_simple, BiasLayer, PositionalEmbedding, ClassToken, MixupToken
from keras_cv_attention_models.mlp_family.mlp_mixer import mlp_block, mixer_block
from keras_cv_attention_models.mlp_family.res_mlp import ChannelAffine
from keras_cv_attention_models.mlp_family.gated_mlp import spatial_gating_block
from keras_cv_attention_models.mlp_family.wave_mlp import phase_aware_token_mixing
from keras_cv_attention_models.levit.levit import MultiHeadPositionalEmbedding, mhsa_with_multi_head_position_and_strides
from keras_cv_attention_models.nfnets.nfnets import ScaledStandardizedConv2D, ZeroInitGain
from keras_cv_attention_models.beit.beit import MultiHeadRelativePositionalEmbedding, HeadInitializer
from keras_cv_attention_models.coatnet.coatnet import mhsa_with_multi_head_relative_position_embedding
from keras_cv_attention_models.cmt.cmt import light_mhsa_with_multi_head_relative_position_embedding
from keras_cv_attention_models.uniformer.uniformer import multi_head_self_attention

CompatibleExtractPatches.__doc__ = """
For issue https://github.com/leondgarse/keras_cv_attention_models/issues/8,
Perform `tf.image.extract_patches` using `Conv2D` for TPU devices. Also for TFLite conversion.

input: `[batch, height, width, channel]`.
output (compressed=True): `[batch, height // strides,  width // strides, height_kernel * width_kernel * channel]`.
output (compressed=False): `[batch, height // strides,  width // strides, height_kernel, width_kernel, channel]`.

Args:
  sizes: could be `tf.image.extract_patches` sizes format `[1, 3, 3, 1]`, or `Conv2D` kernel_size format `3`.
  strides: could be `tf.image.extract_patches` strides format `[1, 2, 2, 1]`, or `Conv2D` strides format `2`.
  rates: could be `tf.image.extract_patches` rates format `[1, 1, 1, 1]`, or `Conv2D` dilation_rate format `1`.
  padding: "VALID" or "SAME", will perform padding in PyTorch way if "SAME".
  compressed: boolean value if compress extracted `height_kernel`, `width_kernel`, `channel` into 1 dimension.
  force_conv: force using `Conv2D` instead of `tf.image.extract_patches`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = tf.ones([1, 64, 27, 192])
>>> print(attention_layers.CompatibleExtractPatches(sizes=3, strides=2)(aa).shape)
# (1, 32, 14, 1728)
>>> print(attention_layers.CompatibleExtractPatches(sizes=3, strides=2, compressed=False)(aa).shape)
# (1, 32, 14, 3, 3, 192)

# `Conv2D` version Performs slower than `extract_patches`.
>>> cc = attention_layers.CompatibleExtractPatches(sizes=3, strides=2, force_conv=True)
>>> cc(aa).shape  # init run
>>> %timeit cc(aa)
# 772 µs ± 6.71 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
>>> %timeit tf.image.extract_patches(aa, [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
# 108 µs ± 153 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
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
>>> tf_patches = attention_layers.CompatibleExtractPatches(kernel_size, strides)(inputs)
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

EvoNormalization.__doc__ = """
Keras implementation of [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py).
Paper [PDF 2004.02967 Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf).

Args:
  nonlinearity: boolean value if apply non-linearity calculation, default True.
  num_groups: groups for paper `EVONORM_S0` mode, value `<=0` for `EVONORM_B0` mode. Dafult -1.
      - EVONORM_B0: nonlinearity=True, num_groups=-1
      - EVONORM_S0: nonlinearity=True, num_groups > 0
      - EVONORM_B0 / EVONORM_S0 linearity: nonlinearity=False, num_groups=-1
      - EVONORM_S0A linearity: nonlinearity=False, num_groups > 0.
        Defined in https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/evo_norm.py#L239.
  zero_gamma: boolean value if init gamma as 0, default False for init as 1.
  momentum: momentum for `EVONORM_B0` mode, default 0.99.
  epsilon: small constant avoid zero dividing, default 0.001.
  data_format: one of ["channels_first", "channels_last"]. Default "auto" for using `K.image_data_format()`.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> # EVONORM_B0
>>> aa = attention_layers.EvoNormalization()
>>> print(aa(tf.ones([2, 32, 32, 48])).shape)
>>> # (2, 32, 32, 48)
>>> print({ii.name.split("/")[-1]: ii.shape for ii in aa.weights})
>>> # {'gamma:0': TensorShape([1, 1, 1, 48]), 'beta:0': TensorShape([1, 1, 1, 48]), 'vv:0': TensorShape([1, 1, 1, 48]), 'moving_variance:0': TensorShape([1, 1, 1, 48])}

>>> # EVONORM_B0 linearity
>>> aa = attention_layers.EvoNormalization(nonlinearity=False)
>>> print(aa(tf.ones([2, 32, 32, 56, 48])).shape)
>>> # (2, 32, 32, 56, 48)
>>> print({ii.name.split("/")[-1]: ii.shape for ii in aa.weights})
>>> # {'gamma:0': TensorShape([1, 1, 1, 1, 48]), 'beta:0': TensorShape([1, 1, 1, 1, 48]), 'moving_variance:0': TensorShape([1, 1, 1, 1, 48])}

>>> # EVONORM_S0, actual using __num_groups__ is made divisible by channel dimension.
>>> aa = attention_layers.EvoNormalization(nonlinearity=True, num_groups=32)
>>> print(aa(tf.ones([2, 32, 56])).shape, aa.__num_groups__) # print actual using __num_groups__
>>> # (2, 32, 56) 28
>>> print({ii.name.split("/")[-1]: ii.shape for ii in aa.weights})
>>> # {'gamma:0': TensorShape([1, 1, 56]), 'beta:0': TensorShape([1, 1, 56]), 'vv:0': TensorShape([1, 1, 56])}

>>> # EVONORM_S0, force "channels_first".
>>> aa = attention_layers.EvoNormalization(nonlinearity=True, num_groups=32, data_format="channels_first")
>>> print(aa(tf.ones([2, 32, 56])).shape, aa.__num_groups__) # print actual using __num_groups__
>>> # (2, 32, 56) 32
>>> print({ii.name.split("/")[-1]: ii.shape for ii in aa.weights})
>>> # {'gamma:0': TensorShape([1, 32, 1]), 'beta:0': TensorShape([1, 32, 1]), 'vv:0': TensorShape([1, 32, 1])}

>>> # EVONORM_S0A linearity
>>> aa = attention_layers.EvoNormalization(nonlinearity=False, num_groups=16)
>>> print(aa(tf.ones([2, 48])).shape)
>>> # (2, 48)
>>> print({ii.name.split("/")[-1]: ii.shape for ii in aa.weights})
>>> # {'gamma:0': TensorShape([1, 48]), 'beta:0': TensorShape([1, 48])}
"""
