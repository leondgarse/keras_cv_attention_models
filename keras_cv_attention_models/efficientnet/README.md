# ___Keras EfficientNetV2___
***

## Summary
  - Merged from [Github leondgarse/keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2).
  - Keras implementation of [Official efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2). Article [arXiv 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le.
  - EfficientNetV2 paper [PDF 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf). Official implementation [Github google/automl/efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2).
  - EfficientNet paper [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf). Official implementation [Github tensorflow/tpu/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
  - `h5` model weights converted from official publication.
  - `effv2-t-imagenet.h5` model weights converted from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models#july-5-9-2021). which claimed both faster and better accuracy than `b3`. Please notice that `PyTorch` using different `bn_epsilon` and `padding` strategy.
***

## Models
  | V2 Model                   | Params | FLOPs  | Input | Top1 Acc | Download                                                                                                                                       |
  | -------------------------- | ------ | ------ | ----- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
  | EfficientNetV2B0           | 7.1M   | 0.72G  | 224   | 78.7     | [effv2b0-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-imagenet.h5) |
  | - ImageNet21k-ft1k         | 7.1M   | 0.72G  | 224   | 77.55?   | [effv2b0-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-21k-ft1k.h5) |
  | EfficientNetV2B1           | 8.1M   | 1.21G  | 240   | 79.8     | [effv2b1-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-imagenet.h5) |
  | - ImageNet21k-ft1k         | 8.1M   | 1.21G  | 240   | 79.03?   | [effv2b1-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-21k-ft1k.h5) |
  | EfficientNetV2B2           | 10.1M  | 1.71G  | 260   | 80.5     | [effv2b2-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-imagenet.h5) |
  | - ImageNet21k-ft1k         | 10.1M  | 1.71G  | 260   | 79.48?   | [effv2b2-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-21k-ft1k.h5) |
  | EfficientNetV2B3           | 14.4M  | 3.03G  | 300   | 82.1     | [effv2b3-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-imagenet.h5) |
  | - ImageNet21k-ft1k         | 14.4M  | 3.03G  | 300   | 82.46?   | [effv2b3-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-21k-ft1k.h5) |
  | EfficientNetV2T            | 13.6M  | 3.18G  | 288   | 82.34    | [effv2t-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-t-imagenet.h5)   |
  | EfficientNetV2T_GC         | 13.7M  | 3.19G  | 288   | 82.46    | [effv2t-gc-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-t-gc-imagenet.h5)   |
  | EfficientNetV2S            | 21.5M  | 8.41G  | 384   | 83.9     | [effv2s-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-imagenet.h5)   |
  | - ImageNet21k-ft1k         | 21.5M  | 8.41G  | 384   | 84.9     | [effv2s-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-21k-ft1k.h5)   |
  | EfficientNetV2M            | 54.1M  | 24.69G | 480   | 85.2     | [effv2m-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-imagenet.h5)   |
  | - ImageNet21k-ft1k         | 54.1M  | 24.69G | 480   | 86.2     | [effv2m-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-21k-ft1k.h5)   |
  | EfficientNetV2L            | 119.5M | 56.27G | 480   | 85.7     | [effv2l-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-imagenet.h5)   |
  | - ImageNet21k-ft1k         | 119.5M | 56.27G | 480   | 86.9     | [effv2l-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-21k-ft1k.h5)   |
  | EfficientNetV2XL, 21k-ft1k | 206.8M | 93.66G | 512   | 87.2     | [effv2xl-21k-ft1k.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-xl-21k-ft1k.h5) |

  - **EfficientNetV1** parameter `include_preprocessing=False` is added, and the default `False` value makes expecting input value in range `[-1, 1]`, same with `EfficientNetV2`. Default `pretrained` is `noisy_student`.

  | V1 Model                       | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------------------------ | ------ | ------- | ----- | -------- | -------- |
  | EfficientNetV1B0               | 5.3M   | 0.39G   | 224   | 77.6     | [effv1-b0-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-imagenet.h5)           |
  | - NoisyStudent                 | 5.3M   | 0.39G   | 224   | 78.8     | [effv1-b0-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-noisy_student.h5) |
  | EfficientNetV1B1               | 7.8M   | 0.70G   | 240   | 79.6     | [effv1-b1-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-imagenet.h5)           |
  | - NoisyStudent                 | 7.8M   | 0.70G   | 240   | 81.5     | [effv1-b1-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-noisy_student.h5) |
  | EfficientNetV1B2               | 9.1M   | 1.01G   | 260   | 80.5     | [effv1-b2-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-imagenet.h5)           |
  | - NoisyStudent                 | 9.1M   | 1.01G   | 260   | 82.4     | [effv1-b2-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-noisy_student.h5) |
  | EfficientNetV1B3               | 12.2M  | 1.86G   | 300   | 81.9     | [effv1-b3-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-imagenet.h5)           |
  | - NoisyStudent                 | 12.2M  | 1.86G   | 300   | 84.1     | [effv1-b3-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-noisy_student.h5) |
  | EfficientNetV1B4               | 19.3M  | 4.46G   | 380   | 83.3     | [effv1-b4-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-imagenet.h5)           |
  | - NoisyStudent                 | 19.3M  | 4.46G   | 380   | 85.3     | [effv1-b4-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-noisy_student.h5) |
  | EfficientNetV1B5               | 30.4M  | 10.40G  | 456   | 84.3     | [effv1-b5-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-imagenet.h5)           |
  | - NoisyStudent                 | 30.4M  | 10.40G  | 456   | 86.1     | [effv1-b5-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-noisy_student.h5) |
  | EfficientNetV1B6               | 43.0M  | 19.29G  | 528   | 84.8     | [effv1-b6-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-imagenet.h5)           |
  | - NoisyStudent                 | 43.0M  | 19.29G  | 528   | 86.4     | [effv1-b6-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-noisy_student.h5) |
  | EfficientNetV1B7               | 66.3M  | 38.13G  | 600   | 85.2     | [effv1-b7-imagenet.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-imagenet.h5)           |
  | - NoisyStudent                 | 66.3M  | 38.13G  | 600   | 86.9     | [effv1-b7-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-noisy_student.h5) |
  | EfficientNetV1L2, NoisyStudent | 480.3M | 477.98G | 800   | 88.4     | [effv1-l2-noisy_student.h5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-l2-noisy_student.h5) |
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
## Verification with Original version
  ```py
  input_shape = 512
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")

  """ Official EfficientNetV2XL """
  sys.path.append('../automl/efficientnetv2/')
  import datasets as orign_datasets
  import effnetv2_model as orign_effnetv2_model

  dataset = "imagenet21k"
  model_type = "xl"
  load_weights = "imagenet21k"
  cc = orign_datasets.get_dataset_config(dataset)
  if cc.get("model", None):
      cc.model.num_classes = cc.data.num_classes
  else:
      cc['model'] = None
  orign_model = orign_effnetv2_model.get_model('efficientnetv2-{}'.format(model_type), model_config=cc.model, weights=load_weights)
  orign_out = orign_model(inputs, training=False)

  """ Keras EfficientNetV2XL """
  from keras_cv_attention_models import efficientnet
  mm = efficientnet.EfficientNetV2XL(pretrained="imagenet21k", classifier_activation=None)
  keras_out = mm(inputs)

  """ Verification """
  print(f"{np.allclose(orign_out, keras_out) = }")
  # np.allclose(orign_out, keras_out) = True
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
