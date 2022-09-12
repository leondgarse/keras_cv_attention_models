# Attention layers
***

## Summary
  - Defined layers / functions from model structures.
  - Basic blocks with some default behavior like `activation_by_name`, `anti_alias_downsample`, `batchnorm_with_activation`, `conv2d_no_bias`, `drop_block`, `layer_norm`, `se_module` from `common_layers`.
  - Layers / blocks defined in model implementations, like `mhsa_with_relative_position_embedding` from `botnet`, `ChannelAffine` from `res_mlp`, `mlp_block` from `mlp_mixer`, which sometimes can be reused.
## Positional Embedding Layers
  - **BiasPositionalEmbedding** from `cmt`. A basic bias layer with `load_resized_weights` method. Positional embedding shape is `[num_heads, query_height * query_width, kv_height * kv_width]`.
  - **ConvPositionalEncoding** from `coat`. Applies a `DepthwiseConv2D` layer with input, then adds with input.
  - **ConvRelativePositionalEncoding** from `coat`. Applies multi `DepthwiseConv2D` layers with split input, then adds with input.
  - **MultiHeadPositionalEmbedding** from `levit`. Positional embedding shape is `[height * width, num_heads]`.
  - **MultiHeadRelativePositionalKernelBias** from `nat`. Positional embedding shape is `[num_heads, (2 * kernel_size - 1) * (2 * kernel_size - 1)]`. It's designed as depending on `num_heads` and `kernel_size`, not on `input_shape`.
  - **MultiHeadRelativePositionalEmbedding** from `beit`. Positional embedding shape is `[num_heads, (2 * height - 1) * (2 * width - 1)]`.
  - **PairWiseRelativePositionalEmbedding** from `swin_transformer_v2`. No weight for this layer, returns a `log` encoded bias depending on input `[height, width]`.
  - **PositionalEmbedding** from `volo`. Positional embedding shape is `[1, height, width, channel]`, then adds directly with input.
  - **PositionalEncodingFourier** from `edgenext`. Layer weight shape depends on parameter `filters` and input channel dimension only, and using `sin` / `cos` encoded distances.
  - **RelativePositionalEmbedding** from `botnet`. Supports both `absolute` / `relative` positional embedding. Layer weights is dotted with input generating positional embedding. It's using same value for all headers.
## Attention Blocks
  - **cot_attention** from `cotnet`. It's using `GroupNormalization` / grouped `Conv2D` / `extract_patches` and other strategies.
  - **cross_covariance_attention** from `edgenext`. It's different from traditional MHSA. This is using `attention_scores` shape `[batch, num_heads, key_dim, key_dim]`, while traditional MHSA `attention_scores` shape `[batch, num_heads, hh * ww, hh * ww]`. Also using cosine distance between `query` and `key` calculating `attention_scores`.
  - **halo_attention** from `halonet`. Extract patches with a `kernel_size` from `key_value` as an enlarged attention area. Also adds `RelativePositionalEmbedding` to `attention_scores`.
  - **light_mhsa_with_multi_head_relative_position_embedding** from `cmt`. Downsample `key_value` with a `sr_ratio` using `DepthwiseConv2D` + `LayerNorm`. Also adds `MultiHeadRelativePositionalEmbedding` to `attention_scores`.
  - **mhsa_with_multi_head_position** from `levit`. Using additional `BatchNormalization` for `query / key / value`, and adding `MultiHeadPositionalEmbedding` to `attention_scores`.
  - **mhsa_with_multi_head_position_and_strides** from `levit`. Using additional `BatchNormalization` for `query / key / value`, and adding `MultiHeadPositionalEmbedding` to `attention_scores`. Also with a `strides` parameter which can further reduce calculation.
  - **mhsa_with_multi_head_relative_position_embedding** from `coatnet`. Typical MHSA with `MultiHeadRelativePositionalEmbedding` added to `attention_scores`.
  - **mhsa_with_relative_position_embedding** from `botnet`. Typical MHSA with `RelativePositionalEmbedding` added to `attention_scores`.
  - **neighborhood_attention** from `nat`. Extract patches with a `kernel_size` from `key_value` as an enlarged attention area. Balancing global and local attention. Also adds `MultiHeadRelativePositionalKernelBias` with `attention_scores`.
  - **multi_head_self_attention** from `uniformer`. Typical multi head self attention block, should work similar with `keras.layers.MultiHeadAttention`.
  - **multi_head_self_attention_channel** from `davit`. It's different from traditional MHSA, that using `attention_scores` shape `[batch, num_heads, key_dim, key_dim]`, while traditional MHSA `attention_scores` shape `[batch, num_heads, hh * ww, hh * ww]`.
  - **outlook_attention** from `volo`. Extract patches with a `kernel_size` from `value` as an enlarged attention area, then matmul with `attention_scores` and fold back.
  - **outlook_attention_simple** from `volo`. Simple version of `outlook_attention` that not using unfold and fold.
  - **shifted_window_attention** from `swin_transformer_v2`. `window_mhsa_with_pair_wise_positional_embedding` with `window_partition` process ahead and `window_reverse` process after. Also supports window shift.
  - **split_attention_conv2d** from `resnest`. Generating `attention_scores` using grouped `Conv2D`.
  - **window_attention** from `davit`. Typical MHSA with `window_partition` process ahead and `window_reverse` process after.
  - **window_mhsa_with_pair_wise_positional_embedding** from `swin_transformer_v2`. Generating `attention_scores` by calculating cosine similarity between `query` and `key`, and applying `PairWiseRelativePositionalEmbedding`.
## Usage Examples
  - **RelativePositionalEmbedding**
    ```py
    from keras_cv_attention_models import attention_layers
    aa = attention_layers.RelativePositionalEmbedding()
    print(f"{aa(tf.ones([1, 4, 14, 16, 256])).shape = }")
    # aa(tf.ones([1, 4, 14, 16, 256])).shape = TensorShape([1, 4, 14, 16, 14, 16])
    ```
  - **outlook_attention**
    ```py
    from keras_cv_attention_models import attention_layers
    inputs = keras.layers.Input([28, 28, 192])
    nn = attention_layers.outlook_attention(inputs, 4, 192)
    cc = keras.models.Model(inputs, nn)
    cc.summary()
    ```
  - **split_attention_conv2d**
    ```py
    from keras_cv_attention_models import attention_layers
    inputs = keras.layers.Input([28, 28, 192])
    nn = attention_layers.split_attention_conv2d(inputs, 384)
    dd = keras.models.Model(inputs, nn)
    dd.summary()
    ```
  - **cot_attention**
    ```py
    from keras_cv_attention_models import attention_layers
    inputs = keras.layers.Input([28, 28, 192])
    nn = attention_layers.cot_attention(inputs, kernel_size=3)
    ee = keras.models.Model(inputs, nn)
    ee.summary()
    ```
***
