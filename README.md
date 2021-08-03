# Keras_cv_attention_models
***

## Usage
  - Install as pip package:
    ```sh
    pip install -U git+https://github.com/leondgarse/keras_cv_attention_models
    ```
    Refer to each sub directory for detail usage.
***
## Layers
  - [attention_layers](keras_cv_attention_models/attention_layers) is `__init__.py` only, which import core layers defined in other model architectures. Like `MHSAWithPositionEmbedding` from `botnet`, `HaloAttention` from `halonet`.
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
## Model surgery
  - [model_surgery](model_surgery) including functions used to change model parameters.
    - `SAMModel`
    - `add_l2_regularizer_2_model`
    - `convert_to_mixed_float16`
    - `convert_mixed_float16_to_float32`
    - `replace_ReLU`
    - `replace_add_with_stochastic_depth`
    - `replace_stochastic_depth_with_add`
## Models
  - [botnet](keras_cv_attention_models/botnet)
    | Model        | params | Image  resolution | Top1 Acc | Download            |
    | ------------ | ------ | ----------------- | -------- | ------------------- |
    | botnet50     | 21M    | 224               | 77.604   | [botnet50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet50.h5)  |

  - [volo](keras_cv_attention_models/volo)
    | Model        | params | Image  resolution | Top1 Acc | Download            |
    | ------------ | ------- | ----------------- | -------- | ------------------- |
    | volo_d1      | 27M     | 224               | 84.2     | [volo_d1_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_224.h5)  |
    | volo_d1 ↑384 | 27M     | 384               | 85.2     | [volo_d1_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_384.h5)  |
    | volo_d2      | 59M     | 224               | 85.2     | [volo_d2_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_224.h5)  |
    | volo_d2 ↑384 | 59M     | 384               | 86.0     | [volo_d2_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_384.h5)  |
    | volo_d3      | 86M     | 224               | 85.4     | [volo_d3_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_224.h5)  |
    | volo_d3 ↑448 | 86M     | 448               | 86.3     | [volo_d3_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_448.h5)  |
    | volo_d4      | 193M    | 224               | 85.7     | [volo_d4_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_224.h5)  |
    | volo_d4 ↑448 | 193M    | 448               | 86.8     | [volo_d4_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_448.h5) |
    | volo_d5      | 296M    | 224               | 86.1     | [volo_d5_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_224.h5) |
    | volo_d5 ↑448 | 296M    | 448               | 87.0     | [volo_d5_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_448.h5) |
    | volo_d5 ↑512 | 296M    | 512               | 87.1     | [volo_d5_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_512.h5) |

  - [resnest](keras_cv_attention_models/resnest)
    | Model          | params | Image  resolution | Top1 Acc | Download            |
    | -------------- | ------ | ----------------- | -------- | ------------------- |
    | resnest50      | 28M    | 224               | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50.h5)  |
    | resnest101     | 49M    | 256               | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101.h5)  |
    | resnest200     | 71M    | 320               | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200.h5)  |
    | resnest269     | 111M   | 416               | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269.h5)  |

  - [halonet](keras_cv_attention_models/halonet)
    | Model   | params | Image resolution | Top1 Acc |
    | ------- | ------ | ---------------- | -------- |
    | halo_b0 | 4.6M   | 256              |          |
    | halo_b1 | 8.8M   | 256              |          |
    | halo_b2 | 11.04M | 256              |          |
    | halo_b3 | 15.1M  | 320              |          |
    | halo_b4 | 31.4M  | 384              | 85.5%    |
    | halo_b5 | 34.4M  | 448              |          |
    | halo_b6 | 47.98M | 512              |          |
    | halo_b7 | 68.4M  | 600              |          |

  - **[In progress]** [CoAtNet](keras_cv_attention_models/coatnet)
  - **[In progress]** [CoTNet](keras_cv_attention_models/cotnet)
***

## Other implemented keras models
  - [Github ypeleg/nfnets-keras](https://github.com/ypeleg/nfnets-keras)
  - [Github faustomorales/vit-keras](https://github.com/faustomorales/vit-keras)
***
