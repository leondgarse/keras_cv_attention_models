# Attention layers
***

## Summary
  - Defined layers / functions from model structures.
  - `cot_attention` from `cotnet`
  - `HaloAttention` from `halonet`
  - `MHSAWithPositionEmbedding` from `botnet`.
  - `outlook_attention` and `outlook_attention_simple` from `volo`.
  - `rsoftmax` and `split_attention_conv2d` from `resnest`
## Usage
  - **MHSAWithPositionEmbedding**
    ```py
    from keras_cv_attention_models import attention_layers
    aa = attention_layers.MHSAWithPositionEmbedding()
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
