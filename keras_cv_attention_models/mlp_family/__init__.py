from keras_cv_attention_models.mlp_family.mlp_mixer import MLPMixer, MLPMixerS32, MLPMixerS16, MLPMixerB32, MLPMixerB16, MLPMixerL32, MLPMixerL16, MLPMixerH14, mlp_block, mixer_block
from keras_cv_attention_models.mlp_family.res_mlp import ResMLP, ResMLP12, ResMLP24, ResMLP36, ResMLP_B24, ChannelAffine
from keras_cv_attention_models.mlp_family.gated_mlp import GMLP, GMLPTiny16, GMLPS16, GMLPB16, spatial_gating_block

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
  dropout: dropout rate if top layers is included.
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
  tokens_mlp_dim: MLP block token level hidden dimenion, where token level means `height * weight` dimention.
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

__resmlp_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of [Github facebookresearch/deit](https://github.com/facebookresearch/deit).
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
