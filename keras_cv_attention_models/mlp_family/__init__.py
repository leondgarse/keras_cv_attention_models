from keras_cv_attention_models.mlp_family.mlp_mixer import MLPMixer, MLPMixerS32, MLPMixerS16, MLPMixerB32, MLPMixerB16, MLPMixerL32, MLPMixerL16, MLPMixerH14, mlp_block, mixer_block
from keras_cv_attention_models.mlp_family.res_mlp import ResMLP, ResMLP12, ResMLP24, ResMLP36, ResMLP_B24, ChannelAffine
from keras_cv_attention_models.mlp_family.gated_mlp import GMLP, GMLPTiny16, GMLPS16, GMLPB16, spatial_gating_block
from keras_cv_attention_models.mlp_family.wave_mlp import WaveMLP, WaveMLP_T, WaveMLP_S, WaveMLP_M, WaveMLP_B, phase_aware_token_mixing

__mlp_mixer_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer#available-mixer-models).
Paper [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
      For `"imagenet21k"` pre-trained model, actual `num_classes` is `21843`.
  activation: activation used in whole model, default `gelu`.
  sam_rho: None zero value to init model using `SAM` training step.
      SAM Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf).
  dropout: top dropout rate if top layers is included.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the `top` layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the `top` layer.
      Default is `softmax`.
  pretrained: value in {pretrained_list}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/`.

Returns:
    A `keras.Model` instance.
"""

MLPMixer.__doc__ = __mlp_mixer_head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  stem_width: stem output channel dimenion.
  tokens_mlp_dim: MLP block token level hidden dimenion, where token level means `height * weight` dimension.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet21k", "imagenet_sam"]) + """
Model architectures:
  | Model       | Params | Top1 Acc | Pre-trained                         |
  | ----------- | ------ | -------- | ----------------------------------- |
  | MLPMixerS32 | 19.1M  | 68.70    | None                                |
  | MLPMixerS16 | 18.5M  | 73.83    | None                                |
  | MLPMixerB32 | 60.3M  | 75.53    | imagenet_sam                        |
  | MLPMixerB16 | 59.9M  | 80.00    | imagenet, imagenet_sam, imagenet21k |
  | MLPMixerL32 | 206.9M | 80.67    | None                                |
  | MLPMixerL16 | 208.2M | 84.82    | imagenet, imagenet21k               |
  | - input 448 | 208.2M | 86.78    | None                                |
  | MLPMixerH14 | 432.3M | 86.32    | None                                |
  | - input 448 | 432.3M | 87.94    | None                                |

  | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
  | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
  | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
  | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
  | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
  | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
  | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
"""


__mixer_default_doc__ = __mlp_mixer_head_doc__ + """
[{model_name} architecture] num_blocks: {num_blocks}, patch_size: {patch_size}, stem_width: {stem_width}, tokens_mlp_dim: {tokens_mlp_dim}, channels_mlp_dim: {channels_mlp_dim}.

Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet21k", "imagenet_sam"])

MLPMixerS32.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerS32", **mlp_mixer.BLOCK_CONFIGS["s32"])
MLPMixerS16.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerS16", **mlp_mixer.BLOCK_CONFIGS["s16"])
MLPMixerB32.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerB32", **mlp_mixer.BLOCK_CONFIGS["b32"])
MLPMixerB16.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerB16", **mlp_mixer.BLOCK_CONFIGS["b16"])
MLPMixerL32.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerL32", **mlp_mixer.BLOCK_CONFIGS["l32"])
MLPMixerL16.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerL16", **mlp_mixer.BLOCK_CONFIGS["l16"])
MLPMixerH14.__doc__ = __mixer_default_doc__.format(model_name="MLPMixerH14", **mlp_mixer.BLOCK_CONFIGS["h14"])

mlp_block.__doc__ = __mlp_mixer_head_doc__ + """
MLP block

Args:
  inputs: Input tensor.
  hidden_dim: Expanded channel dimension for the first `Dense` layer.
  activation: activation applied, default `gelu`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14 * 14, 9])
>>> nn = attention_layers.mlp_block(inputs, 9 * 4)
>>> print(f"{nn.shape = }")
nn.shape = TensorShape([None, 196, 9])

>>> keras.models.Model(inputs, nn).summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 196, 9)]          0
_________________________________________________________________
dense (Dense)                (None, 196, 36)           360
_________________________________________________________________
activation (Activation)      (None, 196, 36)           0
_________________________________________________________________
dense_1 (Dense)              (None, 196, 9)            333
=================================================================
Total params: 693
Trainable params: 693
Non-trainable params: 0
_________________________________________________________________
"""
mixer_block.__doc__ = __mlp_mixer_head_doc__ + """
MLP Mixer block

Args:
  inputs: Input tensor.
  tokens_mlp_dim: hidden_dim for the first mlp_block.
  channels_mlp_dim: hidden_dim for the second mlp_block.
  drop_rate: block dropout rate.
  activation: activation for mlp_block.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14 * 14, 9])
>>> nn = attention_layers.mixer_block(inputs, tokens_mlp_dim=14 * 14, channels_mlp_dim=9 * 4)
>>> print(f"{nn.shape = }")
nn.shape = TensorShape([None, 196, 9])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 196, 9)]     0
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, 196, 9)       18          input_1[0][0]
__________________________________________________________________________________________________
permute (Permute)               (None, 9, 196)       0           layer_normalization[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 9, 196)       38612       permute[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 9, 196)       0           dense[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 9, 196)       38612       activation[0][0]
__________________________________________________________________________________________________
permute_1 (Permute)             (None, 196, 9)       0           dense_1[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 196, 9)       0           permute_1[0][0]
                                                                 input_1[0][0]
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, 196, 9)       18          add[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 196, 36)      360         layer_normalization_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 196, 36)      0           dense_2[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 196, 9)       333         activation_1[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 196, 9)       0           dense_3[0][0]
                                                                 add[0][0]
==================================================================================================
Total params: 77,953
Trainable params: 77,953
Non-trainable params: 0
__________________________________________________________________________________________________
"""

__resmlp_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of [Github facebookresearch/resmlp_models.py](https://github.com/facebookresearch/deit/blob/main/resmlp_models.py).
Paper [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf).
"""

ResMLP.__doc__ = __resmlp_head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  stem_width: stem output channel dimenion.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet22k"]) + """
Model architectures:
  | Model         | Params | Image resolution | Top1 Acc | Pre-trained |
  | ------------- | ------ | ---------------- | -------- | ----------- |
  | ResMLP12      | 15M    | 224              | 77.8     | imagenet    |
  | ResMLP24      | 30M    | 224              | 80.8     | imagenet    |
  | ResMLP36      | 116M   | 224              | 81.1     | imagenet    |
  | ResMLP_B24    | 129M   | 224              | 83.6     | imagenet    |
  | - imagenet22k | 129M   | 224              | 84.4     | imagenet22k |
"""

__resmlp_default_doc__ = __resmlp_head_doc__ + """
[{model_name} architecture] num_blocks: {num_blocks}, patch_size: {patch_size}, stem_width: {stem_width}, channels_mlp_dim: {channels_mlp_dim}.

Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet22k"])

ResMLP12.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP12", **res_mlp.BLOCK_CONFIGS["12"])
ResMLP24.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP24", **res_mlp.BLOCK_CONFIGS["24"])
ResMLP36.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP36", **res_mlp.BLOCK_CONFIGS["36"])
ResMLP_B24.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP_B24", **res_mlp.BLOCK_CONFIGS["b24"])

ChannelAffine.__doc__ = __resmlp_head_doc__ + """
ChannelAffine layer

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ChannelAffine()
>>> print(f"{aa(tf.ones([1, 32, 32, 192])).shape = }")
aa(tf.ones([1, 32, 32, 192])).shape = TensorShape([1, 32, 32, 192])

>>> print({ii.name: ii.shape for ii in aa.weights})
{'channel_affine_1/weight:0': TensorShape([192]),
 'channel_affine_1/bias:0': TensorShape([192])}
"""

__gmlp_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of Gated MLP.
Paper [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).
"""

GMLP.__doc__ = __gmlp_head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  stem_width: stem output channel dimenion.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
    | Model      | Params | Image resolution | Top1 Acc | Pre-trained |
    | ---------- | ------ | ---------------- | -------- | ----------- |
    | GMLPTiny16 | 6M     | 224              | 72.3     | None        |
    | GMLPS16    | 20M    | 224              | 79.6     | imagenet    |
    | GMLPB16    | 73M    | 224              | 81.6     | None        |
"""

__gmlp_default_doc__ = __gmlp_head_doc__ + """
[{model_name} architecture] num_blocks: {num_blocks}, patch_size: {patch_size}, stem_width: {stem_width}, channels_mlp_dim: {channels_mlp_dim}.

Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"])

GMLPTiny16.__doc__ = __gmlp_default_doc__.format(model_name="GMLPTiny16", **gated_mlp.BLOCK_CONFIGS["tiny16"])
GMLPS16.__doc__ = __gmlp_default_doc__.format(model_name="GMLPS16", **gated_mlp.BLOCK_CONFIGS["s16"])
GMLPB16.__doc__ = __gmlp_default_doc__.format(model_name="GMLPB16", **gated_mlp.BLOCK_CONFIGS["b16"])

spatial_gating_block.__doc__ = __gmlp_head_doc__ + """
Spatial Gating Block

input: `[batch, height * width, channel]`.
output: `[batch, height * width, channel // 2]`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14 * 14, 8])
>>> nn = attention_layers.spatial_gating_block(inputs)
>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 196, 8)]     0
__________________________________________________________________________________________________
tf.split (TFOpLambda)           [(None, 196, 4), (No 0           input_1[0][0]
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, 196, 4)       8           tf.split[0][1]
__________________________________________________________________________________________________
permute (Permute)               (None, 4, 196)       0           layer_normalization[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 4, 196)       38612       permute[0][0]
__________________________________________________________________________________________________
permute_1 (Permute)             (None, 196, 4)       0           dense[0][0]
__________________________________________________________________________________________________
multiply (Multiply)             (None, 196, 4)       0           tf.split[0][0]
                                                                 permute_1[0][0]
==================================================================================================
Total params: 38,620
Trainable params: 38,620
Non-trainable params: 0
__________________________________________________________________________________________________
"""

__wavemlp_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of [Github huawei-noah/wavemlp_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch).
Paper [PDF 2111.12294 An Image Patch is a Wave: Quantum Inspired Vision MLP](https://arxiv.org/pdf/2111.12294.pdf).
"""

WaveMLP.__doc__ = __wavemlp_head_doc__ + """
Args:
  num_blocks: number of layers.
  out_channels: output channel for each block.
  stem_width: stem output channel dimenion. Default -1 for out_channels[0]
  mlp_ratios: mlp hidden channel expansion ratios for each block.
  use_downsample_norm: boolean value if stem and down saple blocks using norm layers.
      True for WaveMLP_T and WaveMLP_S, False for WaveMLP_M and WaveMLP_B.
  use_group_norm: boolean value if using GroupNormalization instead of BatchNormalization. False for WaveMLP_T, True for others.
  qkv_bias: boolean value if use bias for `phase_aware_token_mixing` blocks.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
  | Model     | Params | Image resolution | Top1 Acc |
  | --------- | ------ | ---------------- | -------- |
  | WaveMLP_T | 17M    | 224              | 80.9     |
  | WaveMLP_S | 30M    | 224              | 82.9     |
  | WaveMLP_M | 44M    | 224              | 83.3     |
  | WaveMLP_B | 63M    | 224              | 83.6     |
"""

WaveMLP_T.__doc__ = __wavemlp_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"])

WaveMLP_S.__doc__ = WaveMLP_T.__doc__
WaveMLP_M.__doc__ = WaveMLP_T.__doc__
WaveMLP_B.__doc__ = WaveMLP_T.__doc__

phase_aware_token_mixing.__doc__ = __wavemlp_head_doc__ + """
Phase-Aware Token Mixing Block

input: `[batch, height, width, channel]`.
output: `[batch, height, width, out_channel]`.

Args:
  inputs: input tensor.
  out_channel: output channel, default -1 for same with inputs.
  qkv_bias: boolean value if use bias for `height` / `width` / `channel` conv layers, default False.
  output_dropout: dropout rate for output, default 0.
  activation: activation for internal mlp block, default "gelu".

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 14, 8])
>>> nn = attention_layers.phase_aware_token_mixing(inputs, name="aa")
>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
# Total params: 786
# Trainable params: 754
# Non-trainable params: 32

>>> print({ii.name: [list(jj.shape) for jj in ii.weights] for ii in mm.layers if len(ii.weights) != 0})
# {'aa_theta_h_conv': [[1, 1, 8, 8], [8]],
#  'aa_theta_w_conv': [[1, 1, 8, 8], [8]],
#  'aa_theta_h_bn': [[8], [8], [8], [8]],
#  'aa_theta_w_bn': [[8], [8], [8], [8]],
#  'aa_height_conv': [[1, 1, 8, 8]],
#  'aa_width_conv': [[1, 1, 8, 8]],
#  'aa_height_down_conv': [[1, 7, 2, 8]],
#  'aa_width_down_conv': [[7, 1, 2, 8]],
#  'aa_channel_conv': [[1, 1, 8, 8]],
#  'aa_reweight_Conv_0': [[1, 1, 8, 2], [2]],
#  'aa_reweight_Conv_1': [[1, 1, 2, 24], [24]],
#  'aa_out_conv': [[1, 1, 8, 8], [8]]}
"""
