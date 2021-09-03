# Attention layers
***

## Summary
  - Defined layers / functions from model structures.
  - Basic `activation_by_name`, `anti_alias_downsample`, `batchnorm_with_activation`, `conv2d_no_bias`, `drop_block`, `hard_swish`, `layer_norm`, `se_module` from `common_layers`
  - `RelativePositionalEmbedding` and `mhsa_with_relative_position_embedding` from `botnet`.
  - `cot_attention` from `cotnet`.
  - `ConvPositionalEncoding`, `ConvRelativePositionalEncoding`, from `coat`.
  - `halo_attention` from `halonet`.
  - `rsoftmax` and `split_attention_conv2d` from `resnest`.
  - `outlook_attention`, `outlook_attention_simple`, `BiasLayer`, `PositionalEmbedding`, `ClassToken` from `volo`.
  - `mlp_block`, `mixer_block` from `mlp_mixer`.
  - `ChannelAffine` from `res_mlp`.
  - `spatial_gating_block` from `gated_mlp`.
  - `MultiHeadPositionalEmbedding`, `mhsa_with_multi_head_position_and_strides` from `levit`.
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
