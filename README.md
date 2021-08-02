# Keras_attention_models
***

## Usage
  - Install as pip package:
    ```sh
    pip install -U git+https://github.com/leondgarse/keras_attention_models
    ```
    Refer to each sub directory for detail usage.
## Models
  - [botnet](keras_attention_models/botnet)
    | Model        | params | Image  resolution | Top1 Acc | Download            |
    | ------------ | ------ | ----------------- | -------- | ------------------- |
    | botnet50     | 21M    | 224               | 77.604   | [botnet50.h5](https://github.com/leondgarse/keras_attention_models/releases/download/botnet/botnet50.h5)  |

  - [resnest](keras_attention_models/resnest)
    | Model        | #params | Image  resolution | Top1 Acc | Download            |
    | ------------ | ------- | ----------------- | -------- | ------------------- |
    | volo_d1      | 27M     | 224               | 84.2     | [d1_224_84.2.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d1_224.h5)  |
    | volo_d1 ↑384 | 27M     | 384               | 85.2     | [d1_384_85.2.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d1_384.h5)  |
    | volo_d2      | 59M     | 224               | 85.2     | [d2_224_85.2.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d2_224.h5)  |
    | volo_d2 ↑384 | 59M     | 384               | 86.0     | [d2_384_86.0.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d2_384.h5)  |
    | volo_d3      | 86M     | 224               | 85.4     | [d3_224_85.4.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d3_224.h5)  |
    | volo_d3 ↑448 | 86M     | 448               | 86.3     | [d3_448_86.3.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d3_448.h5)  |
    | volo_d4      | 193M    | 224               | 85.7     | [d4_224_85.7.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d4_224.h5)  |
    | volo_d4 ↑448 | 193M    | 448               | 86.8     | [d4_448_86.79.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d4_448.h5) |
    | volo_d5      | 296M    | 224               | 86.1     | [d5_224_86.10.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d5_224.h5) |
    | volo_d5 ↑448 | 296M    | 448               | 87.0     | [d5_448_87.0.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d5_448.h5) |
    | volo_d5 ↑512 | 296M    | 512               | 87.1     | [d5_512_87.07.h5](https://github.com/leondgarse/keras_attention_models/releases/download/volo/volo_d5_512.h5) |

  - [volo](keras_attention_models/volo)
    | Model          | params | Image  resolution | Top1 Acc | Download            |
    | -------------- | ------ | ----------------- | -------- | ------------------- |
    | resnest50      | 28M    | 224               | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_attention_models/releases/download/resnest/resnest50.h5)  |
    | resnest101     | 49M    | 256               | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_attention_models/releases/download/resnest/resnest101.h5)  |
    | resnest200     | 71M    | 320               | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_attention_models/releases/download/resnest/resnest200.h5)  |
    | resnest269     | 111M   | 416               | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_attention_models/releases/download/resnest/resnest269.h5)  |

  - [halonet](keras_attention_models/halonet)
