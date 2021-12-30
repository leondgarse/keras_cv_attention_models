# ___Keras EfficientNetV2___
***

## Summary
  - Merged from [Github leondgarse/keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2).
  - Keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2). Article [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
  - `h5` model weights converted from official publication.
  - `effv2-t-imagenet.h5` model weights converted from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models#july-5-9-2021). which claimed both faster and better accuracy than `b3`. Please notice that `PyTorch` using different `bn_epsilon` and `padding` strategy.

  | V2 Model    | Params | Top1 | Input | ImageNet21K | Imagenet21k-ft1k | Imagenet |
  | ----------- | ------ | ---- | ----- | ----------- | ---------------- | -------- |
  | EffV2B0 | 7.1M  | 78.7 | 224 | [v2b0-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-21k.h5)|[v2b0-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-21k-ft1k.h5)|[v2b0-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-imagenet.h5)|
  | EffV2B1 | 8.1M  | 79.8 | 240 | [v2b1-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-21k.h5)|[v2b1-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-21k-ft1k.h5)|[v2b1-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-imagenet.h5)|
  | EffV2B2 | 10.1M | 80.5 | 260 | [v2b2-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-21k.h5)|[v2b2-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-21k-ft1k.h5)|[v2b2-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-imagenet.h5)|
  | EffV2B3 | 14.4M | 82.1 | 300 | [v2b3-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-21k.h5)|[v2b3-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-21k-ft1k.h5)|[v2b3-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-imagenet.h5)|
  | EffV2T | 13.6M | 82.5 | 320 |  | |[v2t-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-t-imagenet.h5)|
  | EffV2S   | 21.5M | 84.9 | 384 | [v2s-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-21k.h5) |[v2s-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-21k-ft1k.h5)|[v2s-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-imagenet.h5)|
  | EffV2M   | 54.1M | 86.2 | 480 | [v2m-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-21k.h5) |[v2m-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-21k-ft1k.h5)|[v2m-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-imagenet.h5)|
  | EffV2L   | 119.5M| 86.9 | 480 | [v2l-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-21k.h5) |[v2l-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-21k-ft1k.h5)|[v2l-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-imagenet.h5)|
  | EffV2XL  | 206.8M| 87.2 | 512 | [v2xl-21k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-xl-21k.h5)|[v2xl-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-xl-21k-ft1k.h5)|  |

  - **EfficientNetV1 noisy_student models** from [Github tensorflow/tpu/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). Paper [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf). Parameter `include_preprocessing=False` is added, and the default `False` value makes expecting input value in range `[-1, 1]`, same with `EfficientNetV2`. Default `pretrained` is `noisy_student`.

  | V1 Model       | Params  | Top1 | Input | noisy_student | Imagenet |
  | -------------- | ------- | ---- | ----- | ------------- | -------- |
  | EffV1B0    | 5.3M    | 78.8 | 224 | [v1-b0-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-noisy_student.h5) | [v1-b0-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-imagenet.h5) |
  | EffV1B1    | 7.8M    | 81.5 | 240 | [v1-b1-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-noisy_student.h5) | [v1-b1-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-imagenet.h5) |
  | EffV1B2    | 9.1M    | 82.4 | 260 | [v1-b2-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-noisy_student.h5) | [v1-b2-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-imagenet.h5) |
  | EffV1B3    | 12.2M   | 84.1 | 300 | [v1-b3-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-noisy_student.h5) | [v1-b3-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-imagenet.h5) |
  | EffV1B4    | 19.3M   | 85.3 | 380 | [v1-b4-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-noisy_student.h5) | [v1-b4-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-imagenet.h5) |
  | EffV1B5    | 30.4M   | 86.1 | 456 | [v1-b5-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-noisy_student.h5) | [v1-b5-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-imagenet.h5) |
  | EffV1B6    | 43.0M   | 86.4 | 528 | [v1-b6-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-noisy_student.h5) | [v1-b6-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-imagenet.h5) |
  | EffV1B7    | 66.3M   | 86.9 | 600 | [v1-b7-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-noisy_student.h5) | [v1-b7-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-imagenet.h5) |
  | EffV1L2    | 480.3M  | 88.4 | 800 | [v1-l2-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-l2-noisy_student.h5) |  |

  - **Self tested imagenet accuracy**
    - `rescale_mode` `torch` means `(image - [0.485, 0.456, 0.406]) / [[0.229, 0.224, 0.225]]`, `tf` means `(image - 0.5) / 0.5`
    - All `resize_method` is `bicubic`.
    - Some testing detail is not clear, so not exactly matching official reported results.
    - Testing Detail is [EfficientNetV2 self tested imagenet accuracy](https://github.com/leondgarse/keras_efficientnet_v2/discussions/16).

  | model        | input | rescale_mode | central_crop | top 1   | top 5   | Reported top1     |
  | ------------ | ----- | ------------ | ------------ | ------- | ------- | ----------------- |
  | EffV2B0      | 224   | torch        | 0.875        | 0.78748 | 0.94386 | 0.787             |
  | EffV2B1      | 240   | torch        | 0.95         | 0.7987  | 0.94936 | 0.798             |
  | EffV2B2      | 260   | torch        | 0.95         | 0.80642 | 0.95262 | 0.805             |
  | EffV2B3      | 300   | torch        | 0.95         | 0.82098 | 0.95896 | 0.821             |
  | EffV2T       | 320   | torch        | 0.99         | 0.82506 | 0.96228 | 0.823 (input 288) |
  | EffV2S       | 384   | tf           | 0.99         | 0.8386  | 0.967   | 0.839             |
  | EffV2M       | 480   | tf           | 0.99         | 0.8509  | 0.973   | 0.852             |
  | EffV2L       | 480   | tf           | 0.99         | 0.855   | 0.97324 | 0.857             |
  | EffV2S ft1k  | 384   | tf           | 0.99         | 0.84328 | 0.97254 | 0.849             |
  | EffV2M ft1k  | 480   | tf           | 0.99         | 0.85606 | 0.9775  | 0.862             |
  | EffV2L ft1k  | 480   | tf           | 0.99         | 0.86294 | 0.9799  | 0.869             |
  | EffV2XL ft1k | 512   | tf           | 0.99         | 0.86532 | 0.97866 | 0.872             |
## Usage
  - **Define model and load pretrained weights** Parameter `pretrained` is added in value `[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]`, default is `imagenet`. Model input value should be in range `[-1, 1]`.
    ```py
    # Will download and load `imagenet` pretrained weights.
    # Model weight is loaded with `by_name=True, skip_mismatch=True`.
    from keras_cv_attention_models import efficientnet
    model = efficientnet.EfficientNetV2S(pretrained="imagenet")

    # Run prediction
    import tensorflow as tf
    from tensorflow import keras
    from skimage.data import chelsea
    imm = tf.image.resize(chelsea(), model.input_shape[1:3]) # Chelsea the cat
    pred = model(tf.expand_dims(imm / 128. - 1., 0)).numpy()
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.8642886), ('n02123159', 'tiger_cat', 0.030793495), ...]
    ```
    Or download `h5` model and load directly
    ```py
    mm = keras.models.load_model('efficientnetv2-b3-21k-ft1k.h5')
    ```
    For `"imagenet21k"` pre-trained model, actual `num_classes` is `21843`.
  - **Exclude model top layers** by set `num_classes=0`.
    ```py
    from keras_cv_attention_models import efficientnet
    model = efficientnet.EfficientNetV2B0(dropout=1e-6, num_classes=0, pretrained="imagenet21k")
    print(model.output_shape)
    # (None, 7, 7, 1280)

    model.save('efficientnetv2-b0-21k-notop.h5')
    ```
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
    ```py
    from keras_cv_attention_models import efficientnet
    model = efficientnet.EfficientNetV2M(input_shape=(None, None, 3), drop_connect_rate=0.2, num_classes=0, pretrained="imagenet21k-ft1k")

    print(model(np.ones([1, 224, 224, 3])).shape)
    # (1, 7, 7, 1280)
    print(model(np.ones([1, 512, 512, 3])).shape)
    # (1, 16, 16, 1280)
    ```
  - **`include_preprocessing`** set `True` will add pre-processing `Rescale + Normalization` after `Input`. Means using input value in range `[0, 255]`. Default value `False` means in range `[-1, 1]`. Works both for `EfficientNetV2` and `EfficientNetV1`.
    ```py
    from keras_cv_attention_models import efficientnet
    model = efficientnet.EfficientNetV1B4(pretrained="noisy_student", include_preprocessing=True)

    from skimage.data import chelsea
    imm = tf.image.resize(chelsea(), model.input_shape[1:3]) # Chelsea the cat
    pred = model(tf.expand_dims(imm, 0)).numpy()  # value in range [0, 255]
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.68414235), ('n02123159', 'tiger_cat', 0.04486668), ...]
    ```
## Training detail from article
  - **Training configures**, `Eval size` is used as the default `input_shape` for each model type.

    | Model   | Train size | Eval size | Dropout | Randaug | Mixup |
    | ------- | ---------- | --------- | ------- | ------- | ----- |
    | EffV2B0 | 192        | 224       | 0.2     | 0       | 0     |
    | EffV2B1 | 192        | 240       | 0.2     | 0       | 0     |
    | EffV2B2 | 208        | 260       | 0.3     | 0       | 0     |
    | EffV2B3 | 240        | 300       | 0.3     | 0       | 0     |
    | EffV2S  | 300        | 384       | 0.2     | 10      | 0     |
    | EffV2M  | 384        | 480       | 0.3     | 15      | 0.2   |
    | EffV2L  | 384        | 480       | 0.4     | 20      | 0.5   |
    | EffV2XL | 384        | 512       | 0.4     | 20      | 0.5   |

  - EfficientNetV2-S architecture

    | Stage | Operator               | Stride | #Channels | #Layers |
    | ----- | ---------------------- | ------ | --------- | ------- |
    | 0     | Conv3x3                | 2      | 24        | 1       |
    | 1     | Fused-MBConv1, k3x3    | 1      | 24        | 2       |
    | 2     | Fused-MBConv4, k3x3    | 2      | 48        | 4       |
    | 3     | Fused-MBConv4, k3x3    | 2      | 64        | 4       |
    | 4     | MBConv4, k3x3, SE0.25  | 2      | 128       | 6       |
    | 5     | MBConv6, k3x3, SE0.25  | 1      | 160       | 9       |
    | 6     | MBConv6, k3x3, SE0.25  | 2      | 256       | 15      |
    | 7     | Conv1x1 & Pooling & FC | -      | 1280      | 1       |

  - Progressive training settings for EfficientNetV2

    |              | S min | S max | M min | M max | L min | L max |
    | ------------ | ----- | ----- | ----- | ----- | ----- | ----- |
    | Image Size   | 128   | 300   | 128   | 380   | 128   | 380   |
    | RandAugment  | 5     | 15    | 5     | 20    | 5     | 25    |
    | Mixup alpha  | 0     | 0     | 0     | 0.2   | 0     | 0.4   |
    | Dropout rate | 0.1   | 0.3   | 0.1   | 0.4   | 0.1   | 0.5   |

  - Imagenet training detail
    - RMSProp optimizer with decay 0.9 and momentum 0.9
    - batch norm momentum 0.99; weight decay 1e-5
    - Each model is trained for 350 epochs with total batch size 4096
    - Learning rate is first warmed up from 0 to 0.256, and then decayed by 0.97 every 2.4 epochs
    - We use exponential moving average with 0.9999 decay rate
    - RandAugment (Cubuk et al., 2020)
    - Mixup (Zhang et al., 2018)
    - Dropout (Srivastava et al., 2014)
    - and stochastic depth (Huang et al., 2016) with 0.8 survival probability
***
