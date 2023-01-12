from keras_cv_attention_models.beit.beit import (
    Beit,
    BeitBasePatch16,
    BeitLargePatch16,
    BeitV2BasePatch16,
    BeitV2LargePatch16,
    PatchConv2DWithResampleWeights,
    MultiHeadRelativePositionalEmbedding,
    keras_model_load_weights_from_pytorch_model,
)
from keras_cv_attention_models.beit.flexivit import ViT, FlexiViTSmall, FlexiViTBase, FlexiViTLarge
from keras_cv_attention_models.beit.eva import EvaLargePatch14, EvaGiantPatch14

__beit_head_doc__ = """
Keras implementation of [beit](https://github.com/microsoft/unilm/tree/master/beit).
Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).
"""

__beitv2_head_doc__ = """
Keras implementation of [Github microsoft/beit2](https://github.com/microsoft/unilm/tree/master/beit2).
Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).
"""

__vit_head_doc__ = """
Keras implementation of [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer).
Paper [PDF 2010.11929 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf).
"""

__flexivit_head_doc__ = """
Keras implementation of [Github google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/flexivit).
Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).
"""

__eva_head_doc__ = """
Keras implementation of [Github baaivision/EVA](https://github.com/baaivision/EVA).
Paper [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.
  force_reload_mismatch: boolean value, set True if `patch_size` changed, will force reloading `pos_emb` and `stem_conv` weights.

Returns:
    A `keras.Model` instance.
"""

__class_tail_doc__ = """
Args:
  depth: number of blocks. Default `12`.
  embed_dim: channel dimension for stem and all blocks. Default `768`.
  num_heads: heads number for transformer blocks. Default `12`.
  mlp_ratio: dimension expansion ration for `mlp_block`s. Default `4`.
  patch_size: stem patch size. Default `16`.
  attn_key_dim: key dimension for transformer blocks. Default `0`.
  attn_qv_bias: boolean value, if True and attn_qkv_bias being False, will add BiasLayer for `query` and `value` in transformer.
      Default False for Vit, True for Beit.
  attn_qkv_bias: boolean value, if True will just use bias in `qkv_dense`, and set `qv_bias` False.
      Default True for Vit, False for Beit.
  attn_out_weight: boolean value if use output dense for transformer. Default `True`.
  attn_out_bias: boolean value if output dense use bias for transformer. Default `True`.
  attn_dropout: `attention_score` dropout rate. Default `0`.
  gamma_init_value: init value for `attention` and `mlp` branch `gamma`, if > 0 will use `layer_scale` on block output.
      Default 0 for Vit, 0.1 for Beit
  use_abs_pos_emb: boolean value if use abcolute positional embedding or relative one in attention blocks.
      Default True for Vit, False for Beit.
  use_abs_pos_emb_on_cls_token: boolean value, if `use_abs_pos_emb` is True, whether apply `pos_emb` on `cls_token`.
      False for `FlexiViT`, same as `no_embed_class` in timm. Default True for others.
  use_mean_pooling: boolean value if use mean output or `class_token` output. Default False for Vit, True for Beit.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                 | Params  | FLOPs   | Input | Top1 Acc |
  | --------------------- | ------- | ------- | ----- | -------- |
  | BeitBasePatch16, 21k  | 86.53M  | 17.61G  | 224   | 85.240   |
  |                       | 86.74M  | 55.70G  | 384   | 86.808   |
  | BeitLargePatch16, 21k | 304.43M | 61.68G  | 224   | 87.476   |
  |                       | 305.00M | 191.65G | 384   | 88.382   |
  |                       | 305.67M | 363.46G | 512   | 88.584   |

  | Model              | Params  | FLOPs  | Input | Top1 Acc |
  | ------------------ | ------- | ------ | ----- | -------- |
  | BeitV2BasePatch16  | 86.53M  | 17.61G | 224   | 85.5     |
  | - imagenet21k-ft1k | 86.53M  | 17.61G | 224   | 86.5     |
  | BeitV2BasePatch16  | 304.43M | 61.68G | 224   | 87.3     |
  | - imagenet21k-ft1k | 304.43M | 61.68G | 224   | 88.4     |

  | Model         | Params  | FLOPs  | Input | Top1 Acc |
  | ------------- | ------- | ------ | ----- | -------- |
  | FlexiViTSmall | 22.06M  | 5.36G  | 240   | 82.53    |
  | FlexiViTBase  | 86.59M  | 20.33G | 240   | 84.66    |
  | FlexiViTLarge | 304.47M | 71.09G | 240   | 85.64    |

  | Model                 | Params  | FLOPs    | Input | Top1 Acc |
  | --------------------- | ------- | -------- | ----- | -------- |
  | EvaLargePatch14, 22k  | 304.14M | 61.65G   | 196   | 88.59    |
  |                       | 304.53M | 191.55G  | 336   | 89.20    |
  | EvaGiantPatch14, clip | 1012.6M | 267.40G  | 224   | 89.10    |
  | - m30m                | 1013.0M | 621.45G  | 336   | 89.57    |
  | - m30m                | 1014.4M | 1911.61G | 560   | 89.80    |
"""

Beit.__doc__ = __beit_head_doc__ + __class_tail_doc__
ViT.__doc__ = __vit_head_doc__ + __class_tail_doc__

__model_tail_doc__ = """
Args:
""" + __tail_doc__

BeitBasePatch16.__doc__ = __beit_head_doc__ + __model_tail_doc__
BeitLargePatch16.__doc__ = BeitBasePatch16.__doc__

BeitV2BasePatch16.__doc__ = __beitv2_head_doc__ + __model_tail_doc__
BeitV2LargePatch16.__doc__ = BeitV2BasePatch16.__doc__

FlexiViTSmall.__doc__ = __flexivit_head_doc__ + __model_tail_doc__
FlexiViTBase.__doc__ = FlexiViTSmall.__doc__
FlexiViTLarge.__doc__ = FlexiViTSmall.__doc__

EvaLargePatch14.__doc__ = __eva_head_doc__ + __model_tail_doc__
EvaGiantPatch14.__doc__ = EvaLargePatch14.__doc__

MultiHeadRelativePositionalEmbedding.__doc__ = __beit_head_doc__ + """
Multi Head Relative Positional Embedding layer.

Positional embedding shape is `[num_heads, (2 * height - 1) * (2 * width - 1)]`.
input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
output: `[batch, num_heads, attn_blocks, attn_blocks] + positional_bias`.
conditions: attn_height == attn_width

Args:
  with_cls_token: boolean value if input is with class_token.
  attn_height: specify `height` for `attn_blocks` if not square `attn_height != attn_width`.
  num_heads: specify num_heads, or using `input.shape[1]`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MultiHeadRelativePositionalEmbedding()
>>> print(f"{aa(tf.ones([1, 8, 29 * 29 + 1, 29 * 29 + 1])).shape = }")
# aa(tf.ones([1, 8, 29 * 29 + 1, 29 * 29 + 1])).shape = TensorShape([1, 8, 842, 842])
>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_relative_positional_embedding/pos_emb:0': (3252, 8)}

>>> aa = attention_layers.MultiHeadRelativePositionalEmbedding(with_cls_token=False)
>>> print(f"{aa(tf.ones([1, 8, 29 * 29, 29 * 29])).shape = }")
# aa(tf.ones([1, 8, 29 * 29, 29 * 29])).shape = TensorShape([1, 8, 841, 841])
>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_relative_positional_embedding_1/pos_emb:0': (3249, 8)}

>>> plt.imshow(aa.relative_position_index)
"""

PatchConv2DWithResampleWeights.__doc__ = __flexivit_head_doc__ + """
Source implementation https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30
Typical Conv2D layer with `load_resized_weights` function, also using PyTorch padding style.

Args: Same with Conv2D.
"""

keras_model_load_weights_from_pytorch_model.__doc__ = """
Keras ViT model reload weights from timm torch model using `keras_cv_attention_models.download_and_load.keras_reload_from_torch_model`.
Can be used for loading most `Beit` / `ViT` / `FlexViT` / `EVA` models.
Parameters in this using of `keras_reload_from_torch_model` is specifically fitted for timm ViT models.

Args:
  keras_model: keras model same architecture with `timm_vit_model`.
      Need to set parameters for `keras_cv_attention_models.vit.ViT` same as torch model.
  timm_vit_model: built timm vit model.
  save_name: output file name. If ends with `.h5` will save a h5 model, or will be saved_model format.
      Default None for using `{model_name}_{input_shape}.h5`.

Outputs:
  Converted `save_name` weight file, which can be used for afterward and further usage.

Examples:
>>> # Build a timm ViT model
>>> import timm
>>> torch_model = timm.models.vit_tiny_patch16_224(pretrained=True)
>>> _ = torch_model.eval()
>>>
>>> # Build a ViT model same architecture with torch_model
>>> from keras_cv_attention_models import vit
>>> mm = vit.ViT(depth=12, embed_dim=192, num_heads=3, pretrained=None, classifier_activation=None)
>>> vit.keras_model_load_weights_from_pytorch_model(mm, torch_model)
# >>>> Save model to: vit_224.h5
# >>>> Keras model prediction: [('n02123045', 'tabby', 11.990417), ('n02123159', 'tiger_cat', 11.630723), ...]
# >>>> Torch model prediction: [[('n02123045', 'tabby', 11.99042), ('n02123159', 'tiger_cat', 11.630725), ...]
"""
