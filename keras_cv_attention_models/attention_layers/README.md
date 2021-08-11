# Attention layers
***

## Summary
  - Defined layers / functions from model structures.
  - `cot_attention` from `cotnet`.
  - `HaloAttention` from `halonet`.
  - `MHSAWithPositionEmbedding` from `botnet`.
  - `outlook_attention` and `outlook_attention_simple` from `volo`.
  - `rsoftmax` and `split_attention_conv2d` from `resnest`.
  - `groups_depthwise` from `resnext`.
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
## ResNet Models
  - [PDF 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
  - [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf)
    | Model                                                 | params | Top1 Acc |
    | ----------------------------------------------------- | ------ | -------- |
    | Basic ResNet50_V1                                     | 25M    | 76.21    |
    | ResNet50_V1-B (block strides 1, 2 -> 2, 1)            | 25M    | 76.66    |
    | ResNet50_V1-C (deep stem)                             | 25M    | 76.87    |
    | ResNet50_V1-D (block shortcut avgpool + conv)         | 25M    | 77.16    |
    | ResNet50_V1-D + cosine lr decay                       | 25M    | 77.91    |
    | ResNet50_V1-D + cosine + label smoothing 0.1          | 25M    | 78.31    |
    | ResNet50_V1-D + cosine + LS 0.1 + mixup 0.2           | 25M    | 79.15    |
    | ResNet50_V1-D + cosine + LS 0.1 + mixup 0.2 + autoaug | 25M    | 79.41    |
    | ResNeSt-50                                            | 27M    | 81.13    |

  - [PDF 2103.07579 Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/pdf/2103.07579.pdf)
    | Model                      | Top1 Acc | ∆    |
    | -------------------------- | -------- | ---- |
    | ResNet-200                 | 79.0     | —    |
    | + Cosine LR Decay          | 79.3     | +0.3 |
    | + Increase training epochs | 78.8 †   | -0.5 |
    | + EMA of weights           | 79.1     | +0.3 |
    | + Label Smoothing          | 80.4     | +1.3 |
    | + Stochastic Depth         | 80.6     | +0.2 |
    | + RandAugment              | 81.0     | +0.4 |
    | + Dropout on FC            | 80.7 ‡   | -0.3 |
    | + Decrease weight decay    | 82.2     | +1.5 |
    | + Squeeze-and-Excitation   | 82.9     | +0.7 |
    | + ResNet-D                 | 83.4     | +0.5 |

    | Model      | Regularization | Weight Decay 1e-4 | Weight Decay 4e-5 | ∆    |
    | ---------- | -------------- | ----------------- | ----------------- | ---- |
    | ResNet-50  | None           | 79.7              | 78.7              | -1.0 |
    | ResNet-50  | RA-LS          | 82.4              | 82.3              | -0.1 |
    | ResNet-50  | RA-LS-DO       | 82.2              | 82.7              | +0.5 |
    | ResNet-200 | None           | 82.5              | 81.7              | -0.8 |
    | ResNet-200 | RA-LS          | 85.2              | 84.9              | -0.3 |
    | ResNet-200 | RA-LS-SD-DO    | 85.3              | 85.5              | +0.2 |
***
