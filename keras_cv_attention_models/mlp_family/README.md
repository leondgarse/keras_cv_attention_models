# ___Keras MLP___
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Keras_mlp](#kerasmlp)
  - [Usage](#usage)
  - [MLP mixer](#mlp-mixer)
  - [ResMLP](#resmlp)
  - [GMLP](#gmlp)

<!-- /TOC -->
***

## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import mlp_family
    # Will download and load `imagenet` pretrained weights.
    # Model weight is loaded with `by_name=True, skip_mismatch=True`.
    mm = mlp_family.MLPMixerB16(num_classes=1000, pretrained="imagenet")

    # Run prediction
    import tensorflow as tf
    from tensorflow import keras
    from skimage.data import chelsea # Chelsea the cat
    imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='tf') # model="tf" or "torch"
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.9568315), ('n02123045', 'tabby', 0.017994137), ...]
    ```
    For `"imagenet21k"` pre-trained models, actual `num_classes` is `21843`.
  - **Exclude model top layers** by set `num_classes=0`.
    ```py
    from keras_cv_attention_models import mlp_family
    mm = mlp_family.ResMLP_B24(num_classes=0, pretrained="imagenet22k")
    print(mm.output_shape)
    # (None, 784, 768)

    mm.save('resmlp_b24_imagenet22k-notop.h5')
    ```
## MLP mixer
  - [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).
  - [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer#available-mixer-models).
  - **Models** `Top1 Acc` is `Pre-trained on JFT-300M` model accuray on `ImageNet 1K` from paper.
    | Model       | Params | Top1 Acc | ImageNet | Imagenet21k | ImageNet SAM |
    | ----------- | ------ | -------- | --------------- | ------------------ | ------------------- |
    | MLPMixerS32 | 19.1M  | 68.70    |                 |                    |                     |
    | MLPMixerS16 | 18.5M  | 73.83    |                 |                    |                     |
    | MLPMixerB32 | 60.3M  | 75.53    |                 |                    | [b32_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b32_imagenet_sam.h5) |
    | MLPMixerB16 | 59.9M  | 80.00    | [b16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet.h5) | [b16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet21k.h5) | [b16_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet_sam.h5) |
    | MLPMixerL32 | 206.9M | 80.67    |  |  |                     |
    | MLPMixerL16 | 208.2M | 84.82    | [l16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet.h5) | [l16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet21k.h5) |                     |
    | - input 448 | 208.2M | 86.78    |                 |                    |                     |
    | MLPMixerH14 | 432.3M | 86.32    |                 |                    |                     |
    | - input 448 | 432.3M | 87.94    |                 |                    |                     |

    | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
    | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
    | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
    | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
    | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
    | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
    | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
  - Parameter `pretrained` is added in value `[None, "imagenet", "imagenet21k", "imagenet_sam"]`. Default is `imagenet`.
  - **Pre-training details**
    - We pre-train all models using Adam with β1 = 0.9, β2 = 0.999, and batch size 4 096, using weight decay, and gradient clipping at global norm 1.
    - We use a linear learning rate warmup of 10k steps and linear decay.
    - We pre-train all models at resolution 224.
    - For JFT-300M, we pre-process images by applying the cropping technique from Szegedy et al. [44] in addition to random horizontal flipping.
    - For ImageNet and ImageNet-21k, we employ additional data augmentation and regularization techniques.
    - In particular, we use RandAugment [12], mixup [56], dropout [42], and stochastic depth [19].
    - This set of techniques was inspired by the timm library [52] and Touvron et al. [46].
    - More details on these hyperparameters are provided in Supplementary B.
## ResMLP
  - [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)
  - [Github facebookresearch/deit](https://github.com/facebookresearch/deit)
  - **Models** reloaded `imagenet` weights are the `distilled` version from official.
    | Model      | Params | Image resolution | Top1 Acc | Download |
    | ---------- | ------ | ---------------- | -------- | -------- |
    | ResMLP12   | 15M    | 224              | 77.8     | [resmlp12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp12_imagenet.h5) |
    | ResMLP24   | 30M    | 224              | 80.8     | [resmlp24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp24_imagenet.h5) |
    | ResMLP36   | 116M   | 224              | 81.1     | [resmlp36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp36_imagenet.h5) |
    | ResMLP_B24 | 129M   | 224              | 83.6     | [resmlp_b24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet.h5) |
    | - imagenet22k | 129M   | 224              | 84.4     | [resmlp_b24_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet22k.h5) |

  - Parameter `pretrained` is added in value `[None, "imagenet", "imagenet22k"]`, where `imagenet22k` means pre-trained on `imagenet21k` and fine-tuned on `imagenet`. Default is `imagenet`.
## GMLP
  - [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).
  - Model weights reloaded from [Github timm/models/mlp_mixer](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py).
  - **Models**
    | Model      | Params | Image resolution | Top1 Acc | Download |
    | ---------- | ------ | ---------------- | -------- | -------- |
    | GMLPTiny16 | 6M     | 224              | 72.3     |          |
    | GMLPS16    | 20M    | 224              | 79.6     | [gmlp_s16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/gmlp_s16_imagenet.h5) |
    | GMLPB16    | 73M    | 224              | 81.6     |          |

  - Parameter `pretrained` is added in value `[None, "imagenet"]`. Default is `imagenet`.
***
