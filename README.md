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
  - [attention_layers](keras_cv_attention_models/attention_layers) is `__init__.py` only, which imports core layers defined in model architectures. Like `MHSAWithPositionEmbedding` from `botnet`, `HaloAttention` from `halonet`.
    ```py
    from keras_cv_attention_models import attention_layers
    aa = attention_layers.MHSAWithPositionEmbedding(num_heads=4, key_dim=128, relative=True)
    print(f"{aa(tf.ones([1, 14, 16, 256])).shape = }")
    # aa(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])
    ```
## Model surgery
  - [model_surgery](keras_cv_attention_models/model_surgery) including functions used to change model parameters after built.
    ```py
    from keras_cv_attention_models import model_surgery
    # Replace all ReLU with PReLU
    mm = model_surgery.replace_ReLU(keras.applications.ResNet50(), target_activation='PReLU')
    ```
## Models
  - Basic usage
    ```py
    from keras_cv_attention_models import volo
    mm = volo.VOLO_d1(pretrained="imagenet")

    """ Run predict """
    from skimage.data import chelsea
    img = chelsea() # Chelsea the cat
    imm = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.9692954),
    #  ('n02123045', 'tabby', 0.020203391),
    #  ('n02123159', 'tiger_cat', 0.006867502),
    #  ('n02127052', 'lynx', 0.00017674894),
    #  ('n02123597', 'Siamese_cat', 4.9493494e-05)]
    ```
  - [aotnet](keras_cv_attention_models/aotnet) is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer.
    ```py
    # Mixing se and outlook and halo and mhsa and cot_attention, 21M parameters
    # 50 is just a picked number that larger than the relative `num_block`
    from keras_cv_attention_models import aotnet
    attn_types = [None, "outlook", ["mhsa", "halo"] * 50, "cot"]
    se_ratio = [0.25, 0, 0, 0]
    mm = aotnet.AotNet50V2(attn_types=attn_types, se_ratio=se_ratio, deep_stem=True, strides=1)
    ```
  - [botnet](keras_cv_attention_models/botnet)
    | Model        | Params | Image  resolution | Top1 Acc | Download            |
    | ------------ | ------ | ----------------- | -------- | ------------------- |
    | botnet50     | 21M    | 224               | 77.604   | [botnet50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet50.h5)  |
    | botnet101    | 41M    | 224               |          |  |
    | botnet152    | 56M    | 224               |          |  |

  - [volo](keras_cv_attention_models/volo)
    | Model        | Params | Image  resolution | Top1 Acc | Download            |
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
    | Model          | Params | Image  resolution | Top1 Acc | Download            |
    | -------------- | ------ | ----------------- | -------- | ------------------- |
    | resnest50      | 28M    | 224               | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50.h5)  |
    | resnest101     | 49M    | 256               | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101.h5)  |
    | resnest200     | 71M    | 320               | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200.h5)  |
    | resnest269     | 111M   | 416               | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269.h5)  |

  - [resnext](keras_cv_attention_models/resnext)
    | Model          | Params | Image  resolution | Top1 Acc | Download            |
    | -------------- | ------ | ----------------- | -------- | ------------------- |
    | resnext50      | 25M    | 224               | 77.74    | [resnext50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext50.h5)  |
    | resnext101     | 42M    | 224               | 78.73    | [resnext101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext101.h5)  |

  - [halonet](keras_cv_attention_models/halonet)
    | Model   | Params | Image resolution | Top1 Acc |
    | ------- | ------ | ---------------- | -------- |
    | halo_b0 | 4.6M   | 256              |          |
    | halo_b1 | 8.8M   | 256              |          |
    | halo_b2 | 11.04M | 256              |          |
    | halo_b3 | 15.1M  | 320              |          |
    | halo_b4 | 31.4M  | 384              | 85.5    |
    | halo_b5 | 34.4M  | 448              |          |
    | halo_b6 | 47.98M | 512              |          |
    | halo_b7 | 68.4M  | 600              |          |

  - [CoTNet](keras_cv_attention_models/cotnet)
    | Model          | Params | Image resolution | FLOPs | Top1 Acc | Download            |
    | -------------- |:------:| ---------------- | ----- |:--------:| ------------------- |
    | CoTNet-50      | 22.2M  | 224              | 3.3   |   81.3   | [cotnet50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224.h5) |
    | CoTNeXt-50     | 30.1M  | 224              | 4.3   |   82.1   |  |
    | SE-CoTNetD-50  | 23.1M  | 224              | 4.1   |   81.6   | [se_cotnetd50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd50_224.h5) |
    | CoTNet-101     | 38.3M  | 224              | 6.1   |   82.8   | [cotnet101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224.h5) |
    | CoTNeXt-101    | 53.4M  | 224              | 8.2   |   83.2   |  |
    | SE-CoTNetD-101 | 40.9M  | 224              | 8.5   |   83.2   | [se_cotnetd101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd101_224.h5) |
    | SE-CoTNetD-152 | 55.8M  | 224              | 17.0  |   84.0   | [se_cotnetd152_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_224.h5) |
    | SE-CoTNetD-152 | 55.8M  | 320              | 26.5  |   84.6   | [se_cotnetd152_320.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_320.h5) |

  - [CoAtNet](keras_cv_attention_models/coatnet)
    | Model                                | Params | Image resolution | Top1 Acc |
    | ------------------------------------ | ------ | ---------------- | -------- |
    | CoAtNet-0                            | 25M    | 224              | 81.6     |
    | CoAtNet-1                            | 42M    | 224              | 83.3     |
    | CoAtNet-2                            | 75M    | 224              | 84.1     |
    | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
    | CoAtNet-3                            | 168M   | 224              | 84.5     |
    | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
    | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
    | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
    | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |
***

## Other implemented keras models
  - [Github ypeleg/nfnets-keras](https://github.com/ypeleg/nfnets-keras)
  - [Github faustomorales/vit-keras](https://github.com/faustomorales/vit-keras)
  - [Github tensorflow/resnet_rs](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs)
***
