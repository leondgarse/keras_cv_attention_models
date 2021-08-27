# Attention layers
***

## Summary
  - Defined layers / functions from model structures.
  - Basic `anti_alias_downsample`, `batchnorm_with_activation`, `conv2d_no_bias`, `se_module` from `aotnet`
  - `MHSAWithPositionEmbedding` from `botnet`.
  - `cot_attention` from `cotnet`.
  - `ConvPositionalEncoding`, `ConvRelativePositionalEncoding`, `layer_norm` from `coat`.
  - `HaloAttention` from `halonet`.
  - `rsoftmax` and `split_attention_conv2d` from `resnest`.
  - `outlook_attention`, `outlook_attention_simple`, `BiasLayer`, `PositionalEmbedding`, `ClassToken` from `volo`.
  - `mlp_block`, `mixer_block` from `mlp_mixer`.
  - `ChannelAffine` from `res_mlp`.
  - `spatial_gating_block` from `gated_mlp`.
## Usage
  - **MHSAWithPositionEmbedding**
    ```py
    from keras_cv_attention_models import attention_layers
    aa = attention_layers.MHSAWithPositionEmbedding(num_heads=4, key_dim=128)
    print(f"{aa(tf.ones([1, 14, 16, 256])).shape = }")
    # aa(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])

    print({ii.name:ii.numpy().shape for ii in aa.weights})
    # {'mhsa_with_position_embedding_2/query:0': (256, 512),
    #  'mhsa_with_position_embedding_2/key:0': (256, 512),
    #  'mhsa_with_position_embedding_2/value:0': (256, 512),
    #  'mhsa_with_position_embedding_2/output_weight:0': (512, 512),
    #  'mhsa_with_position_embedding_2/r_width:0': (128, 31),
    #  'mhsa_with_position_embedding_2/r_height:0': (128, 27)}
    ```
  - **HaloAttention**
    ```py
    from keras_cv_attention_models import attention_layers
    bb = attention_layers.HaloAttention()
    print(f"{bb(tf.ones([1, 14, 16, 256])).shape = }")
    # bb(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])

    print({ii.name:ii.numpy().shape for ii in bb.weights})
    # {'halo_attention_2/query:0': (256, 512),
    #  'halo_attention_2/key_value:0': (256, 1024),
    #  'halo_attention_2/output_weight:0': (512, 512),
    #  'halo_attention_2/r_width:0': (128, 7),
    #  'halo_attention_2/r_height:0': (128, 7)}
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
