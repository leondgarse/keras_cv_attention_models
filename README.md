# Keras_cv_attention_models
***

## Usage
  - Install as pip package:
    ```sh
    pip install -U git+https://github.com/leondgarse/keras_cv_attention_models
    ```
    Refer to each sub directory for detail usage.
***

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
    | Model                                | params | Image resolution | Top1 Acc |
    | ------------------------------------ | ------ | ---------------- | -------- |
    | CoAtNet-0                            | 25M    | 224              | 81.6     |
    | CoAtNet-0                            | 25M    | 384              | 83.9     |
    | CoAtNet-1                            | 42M    | 224              | 83.3     |
    | CoAtNet-1                            | 42M    | 384              | 85.1     |
    | CoAtNet-2                            | 75M    | 224              | 84.1     |
    | CoAtNet-2                            | 75M    | 384              | 85.7     |
    | CoAtNet-2                            | 75M    | 512              | 85.9     |
    | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
    | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 384              | 87.1     |
    | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 512              | 87.3     |
    | CoAtNet-3                            | 168M   | 224              | 84.5     |
    | CoAtNet-3                            | 168M   | 384              | 85.8     |
    | CoAtNet-3                            | 168M   | 512              | 86.0     |
    | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
    | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 384              | 87.6     |
    | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
    | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 384              | 87.9     |
    | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
    | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 384              | 88.4     |
    | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |

  - **[In progress]** [CoTNet](keras_cv_attention_models/cotnet)
***

## Other implemented keras models
  - [Github ypeleg/nfnets-keras](https://github.com/ypeleg/nfnets-keras)
  - [Github faustomorales/vit-keras](https://github.com/faustomorales/vit-keras)
***
