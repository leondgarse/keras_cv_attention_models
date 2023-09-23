from keras_cv_attention_models.beit.beit import (
    Beit,
    BeitBasePatch16,
    BeitLargePatch16,
    BeitV2BasePatch16,
    BeitV2LargePatch16,
    PatchConv2DWithResampleWeights,
    MultiHeadRelativePositionalEmbedding,
    PositionalEncodingFourierRot,
    keras_model_load_weights_from_pytorch_model,
)
from keras_cv_attention_models.beit.dinov2 import DINOv2, DINOv2_ViT_Small14, DINOv2_ViT_Base14, DINOv2_ViT_Large14, DINOv2_ViT_Giant14
from keras_cv_attention_models.beit.eva import EVA, EvaLargePatch14, EvaGiantPatch14
from keras_cv_attention_models.beit.eva02 import EVA02, EVA02TinyPatch14, EVA02SmallPatch14, EVA02BasePatch14, EVA02LargePatch14
from keras_cv_attention_models.beit.flexivit import FlexiViT, FlexiViTSmall, FlexiViTBase, FlexiViTLarge
from keras_cv_attention_models.beit.meta_transformer import MetaTransformer, MetaTransformerBasePatch16, MetaTransformerLargePatch14
from keras_cv_attention_models.beit.vit import ViT, ViTTinyPatch16, ViTBasePatch16, ViTLargePatch14, ViTText, ViTTextLargePatch14

__beit_head_doc__ = """
Keras implementation of [beit](https://github.com/microsoft/unilm/tree/master/beit).
Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).
"""

__beitv2_head_doc__ = """
Keras implementation of [Github microsoft/beit2](https://github.com/microsoft/unilm/tree/master/beit2).
Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).
"""

__dinov2_head_doc__ = """
Keras implementation of [Github facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
Paper [PDF 2304.07193 DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf).
"""

__eva_head_doc__ = """
Keras implementation of [Github baaivision/EVA](https://github.com/baaivision/EVA).
Paper [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf).
"""

__eva02_head_doc__ = """
Keras implementation of [Github baaivision/EVA/EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02).
Paper [PDF 2303.11331 EVA: EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/pdf/2303.11331.pdf).
"""

__flexivit_head_doc__ = """
Keras implementation of [Github google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/flexivit).
Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).
"""

__vit_head_doc__ = """
Keras implementation of [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer).
Paper [PDF 2010.11929 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf).
"""

__meta_transformer_head_doc__ = """
Keras implementation of [Github invictus717/MetaTransformer](https://github.com/invictus717/MetaTransformer).
Paper [PDF 2307.10802 Meta-Transformer: A Unified Framework for Multimodal Learning](https://arxiv.org/abs/2307.10802).
"""

__tail_doc__ = """  patch_size: stem patch size. Default {patch_size}.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  layer_scale: init value for `attention` and `mlp` branch `gamma`, if > 0 will use `layer_scale` on block output.
      Default 0 for Vit and EVA, 0.1 for Beit, 1.0 for DINOv2.
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
  model_name: string, model name.

  [Attention args]
  attn_key_dim: key dimension for transformer blocks. Default `0`.
  attn_qv_bias: boolean value, if True and attn_qkv_bias being False, will add BiasLayer for `query` and `value` in transformer.
      Default False for Vit, True for Beit.
  attn_qkv_bias: boolean value, if True will just use bias in `qkv_dense`, and set `qv_bias` False.
      Default True for Vit, False for Beit.
  attn_out_weight: boolean value if use output dense for transformer. Default `True`.
  attn_out_bias: boolean value if output dense use bias for transformer. Default `True`.
  attn_dropout: `attention_score` dropout rate. Default `0`.
  use_abs_pos_emb: boolean value if use abcolute positional embedding or relative one in attention blocks.
      Default True for Vit, False for Beit.
  use_abs_pos_emb_on_cls_token: boolean value, if `use_abs_pos_emb` is True, whether apply `pos_emb` on `cls_token`.
      False for `FlexiViT`, same as `no_embed_class` in timm. Default True for others.
  use_rot_pos_emb: boolean value if use `PositionalEncodingFourierRot` on attention query and key.
      True for EVA02, False for others.

  [MLP args]
  mlp_ratio: dimension expansion ration for `mlp_block`s. Default `4`.
  use_gated_mlp: boolean value if using `dense gated + swish` instead of `activation` in `mlp` block.
      True for DINOv2 and EVA02, False for others.
  use_norm_mlp: boolean value if use additional LayerNorm for MLP block. True for EVA02 base and large, False for others.

  [Head args]
  use_mean_pooling_head: boolean value if use mean output or `class_token` output. Default False for Vit, True for Beit.
  use_cat_head: boolean value if use mean output concatenated with `class_token` output. Default True for DINOv2, False for others.

  [Text model args]
  vocab_size: set value > 0 for building text model. Default 49408 for ViTText, 0 for others.
  max_block_size: max block size, works only if vocab_size > 0. Default 77.
  text_positional_dropout: dropout for text model embedding layers. Default 0.
  text_use_positional_embedding: boolean value if use Embedding positional layer after inputs. Default True.
  include_top: boolean value if include top output Dense layer, True for using output channles == vocab_size. Default True.

  [common args]
""" + __tail_doc__

Beit.__doc__ = __beit_head_doc__ + __class_tail_doc__.format(patch_size=16) + """
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
  | BeitV2LargePatch16 | 304.43M | 61.68G | 224   | 87.3     |
  | - imagenet21k-ft1k | 304.43M | 61.68G | 224   | 88.4     |
"""

DINOv2.__doc__ = __dinov2_head_doc__ + __class_tail_doc__.format(patch_size=14) + """
Model architectures:
  | Model              | Params  | FLOPs   | Input | Top1 Acc |
  | ------------------ | ------- | ------- | ----- | -------- |
  | DINOv2_ViT_Small14 | 22.83M  | 47.23G  | 518   | 81.1     |
  | DINOv2_ViT_Base14  | 88.12M  | 152.6G  | 518   | 84.5     |
  | DINOv2_ViT_Large14 | 306.4M  | 509.6G  | 518   | 86.3     |
  | DINOv2_ViT_Giant14 | 1139.6M | 1790.3G | 518   | 86.5     |
"""

EVA.__doc__ = __eva_head_doc__ + __class_tail_doc__.format(patch_size=14) + """
Model architectures:
  | Model                 | Params  | FLOPs    | Input | Top1 Acc |
  | --------------------- | ------- | -------- | ----- | -------- |
  | EvaLargePatch14, 22k  | 304.14M | 61.65G   | 196   | 88.59    |
  |                       | 304.53M | 191.55G  | 336   | 89.20    |
  | EvaGiantPatch14, clip | 1012.6M | 267.40G  | 224   | 89.10    |
  | - m30m                | 1013.0M | 621.45G  | 336   | 89.57    |
  | - m30m                | 1014.4M | 1911.61G | 560   | 89.80    |
"""

EVA02.__doc__ = __eva02_head_doc__ + __class_tail_doc__.format(patch_size=14) + """
Model architectures:
  | Model                                  | Params  | FLOPs   | Input | Top1 Acc |
  | -------------------------------------- | ------- | ------- | ----- | -------- |
  | EVA02TinyPatch14, mim_in22k_ft1k       | 5.76M   | 4.72G   | 336   | 80.658   |
  | EVA02SmallPatch14, mim_in22k_ft1k      | 22.13M  | 15.57G  | 336   | 85.74    |
  | EVA02BasePatch14, mim_in22k_ft22k_ft1k | 87.12M  | 107.6G  | 448   | 88.692   |
  | EVA02LargePatch14, mim_m38m_ft22k_ft1k | 305.08M | 363.68G | 448   | 90.054   |
"""

FlexiViT.__doc__ = __flexivit_head_doc__ + __class_tail_doc__.format(patch_size=16) + """
Model architectures:
  | Model         | Params  | FLOPs  | Input | Top1 Acc |
  | ------------- | ------- | ------ | ----- | -------- |
  | FlexiViTSmall | 22.06M  | 5.36G  | 240   | 82.53    |
  | FlexiViTBase  | 86.59M  | 20.33G | 240   | 84.66    |
  | FlexiViTLarge | 304.47M | 71.09G | 240   | 85.64    |
"""

ViT.__doc__ = __vit_head_doc__ + __class_tail_doc__.format(patch_size=16)
ViTText.__doc__ = __vit_head_doc__ + __class_tail_doc__.format(patch_size=16)

MetaTransformer.__doc__ = __meta_transformer_head_doc__ + __class_tail_doc__.format(patch_size=16) + """
Model architectures:
  | Model                                 | Params  | FLOPs  | Input | Top1 Acc |
  | ------------------------------------- | ------- | ------ | ----- | -------- |
  | MetaTransformerBasePatch16, laion_2b  | 86.86M  | 55.73G | 384   | 85.4     |
  | MetaTransformerLargePatch14, laion_2b | 304.53M | 191.6G | 336   | 88.1     |
"""

__model_tail_doc__ = """
Args:
""" + __tail_doc__

BeitBasePatch16.__doc__ = __beit_head_doc__ + __model_tail_doc__.format(patch_size=16)
BeitLargePatch16.__doc__ = BeitBasePatch16.__doc__

BeitV2BasePatch16.__doc__ = __beitv2_head_doc__ + __model_tail_doc__.format(patch_size=16)
BeitV2LargePatch16.__doc__ = BeitV2BasePatch16.__doc__

DINOv2_ViT_Small14.__doc__ = __dinov2_head_doc__ + __model_tail_doc__.format(patch_size=14)
DINOv2_ViT_Base14.__doc__ = DINOv2_ViT_Small14.__doc__
DINOv2_ViT_Large14.__doc__ = DINOv2_ViT_Small14.__doc__
DINOv2_ViT_Giant14.__doc__ = DINOv2_ViT_Small14.__doc__

EvaLargePatch14.__doc__ = __eva_head_doc__ + __model_tail_doc__.format(patch_size=14)
EvaGiantPatch14.__doc__ = EvaLargePatch14.__doc__

EVA02TinyPatch14.__doc__ = __eva02_head_doc__ + __model_tail_doc__.format(patch_size=14)
EVA02SmallPatch14.__doc__ = EVA02TinyPatch14.__doc__
EVA02BasePatch14.__doc__ = EVA02TinyPatch14.__doc__
EVA02LargePatch14.__doc__ = EVA02TinyPatch14.__doc__

FlexiViTSmall.__doc__ = __flexivit_head_doc__ + __model_tail_doc__.format(patch_size=16)
FlexiViTBase.__doc__ = FlexiViTSmall.__doc__
FlexiViTLarge.__doc__ = FlexiViTSmall.__doc__

MetaTransformerBasePatch16.__doc__ = __meta_transformer_head_doc__ + __model_tail_doc__.format(patch_size=16)
MetaTransformerLargePatch14.__doc__ = __meta_transformer_head_doc__ + __model_tail_doc__.format(patch_size=14)

ViTTinyPatch16.__doc__ = __vit_head_doc__ + __model_tail_doc__.format(patch_size=16)
ViTBasePatch16.__doc__ = __vit_head_doc__ + __model_tail_doc__.format(patch_size=16)
ViTLargePatch14.__doc__ = __vit_head_doc__ + __model_tail_doc__.format(patch_size=14)

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

PositionalEncodingFourierRot.__doc__ = __eva02_head_doc__ + """
Positional Encoding Fourier Rot. No weight.

Positional embedding shape is `pos_sin = pos_cos = [attn_height * attn_width, channels]`.
input (with_cls_token=True): `[batch, ...,  attn_height * attn_width + class_token, channels]`.
input (with_cls_token=False): `[batch, ..., attn_height * attn_width, channels]`.
output: `input * pos_cos + rot * pos_sin`.

Args:
  with_cls_token: boolean value if input is with class_token.
  attn_height: specify `height` for `attn_blocks` if not square `attn_height != attn_width`.
  num_heads: specify num_heads.
      - if <=0, will calculate pos_sin and pos_cos depends on `inputs[-1]`. Default behavior.
      - if > 0, will calculate pos_sin and pos_cos depends on `inputs[-1] // num_heads`,
        then repeat and reshape to match `inputs[-1]`.
  temperature: temperature.
  ref_feature_shape: reference feature shape for resize / fine-tune

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.PositionalEncodingFourierRot()
>>> print(f"{aa(tf.ones([1, 29 * 29 + 1, 3 * 64])).shape = }")
# aa(tf.ones([1, 29 * 29 + 1, 3 * 64])).shape = TensorShape([1, 842, 192])
>>> print(f"{aa.pos_sin.shape = }, {aa.pos_cos.shape = }")
# aa.pos_sin.shape = TensorShape([841, 192]), aa.pos_cos.shape = TensorShape([841, 192])

# with_cls_token=False
>>> aa = attention_layers.PositionalEncodingFourierRot(with_cls_token=False)
>>> print(f"{aa(tf.ones([1, 8, 29 * 29, 3 * 64])).shape = }")
# aa(tf.ones([1, 8, 29 * 29, 3 * 64])).shape = TensorShape([1, 8, 841, 192])
>>> print(f"{aa.pos_sin.shape = }, {aa.pos_cos.shape = }")
# aa.pos_sin.shape = TensorShape([841, 192]), aa.pos_cos.shape = TensorShape([841, 192])

# with_cls_token=False, num_heads=3, pos_sin and pos_cos is repeated 3 times.
>>> aa = attention_layers.PositionalEncodingFourierRot(with_cls_token=False, num_heads=3)
>>> print(f"{aa(tf.ones([1, 8, 29 * 29, 3 * 64])).shape = }")
# aa(tf.ones([1, 8, 29 * 29, 3 * 64])).shape = TensorShape([1, 8, 841, 192])
>>> print(f"{aa.pos_sin.shape = }, {aa.pos_cos.shape = }")
# aa.pos_sin.shape = TensorShape([841, 192]), aa.pos_cos.shape = TensorShape([841, 192])
>>> print(f"{np.allclose(aa.pos_sin[:, :64], aa.pos_sin[:, -64:]) = }")
# np.allclose(aa.pos_sin[:, :64], aa.pos_sin[:, -64:]) = True
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
