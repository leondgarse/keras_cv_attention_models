# ___Keras MLP___
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Keras_mlp](#kerasmlp)
  - [Usage](#usage)
  - [MLP mixer](#mlp-mixer)
  - [ResMLP](#resmlp)
  - [GMLP](#gmlp)
  - [WaveMLP](#wavemlp)

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
  - **Models**
    | Model            | Params | FLOPs   | Input | Top1 Acc | Download |
    | ---------------- | ------ | ------- | ----- | -------- | -------- |
    | MLPMixerS32, JFT | 19.1M  | 1.01G   | 224   | 68.70    |          |
    | MLPMixerS16, JFT | 18.5M  | 3.79G   | 224   | 73.83    |          |
    | MLPMixerB32, JFT | 60.3M  | 3.25G   | 224   | 75.53    |          |
    | - imagenet_sam   | 60.3M  | 3.25G   | 224   | 72.47    | [b32_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b32_imagenet_sam.h5) |
    | MLPMixerB16      | 59.9M  | 12.64G  | 224   | 76.44    | [b16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet.h5) |
    | - imagenet21k    | 59.9M  | 12.64G  | 224   | 80.64    | [b16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet21k.h5) |
    | - imagenet_sam   | 59.9M  | 12.64G  | 224   | 77.36    | [b16_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet_sam.h5) |
    | - JFT            | 59.9M  | 12.64G  | 224   | 80.00    |          |
    | MLPMixerL32, JFT | 206.9M | 11.30G  | 224   | 80.67    |          |
    | MLPMixerL16      | 208.2M | 44.66G  | 224   | 71.76    | [l16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet.h5) |
    | - imagenet21k    | 208.2M | 44.66G  | 224   | 82.89    | [l16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet21k.h5) |
    | - input 448      | 208.2M | 178.54G | 448   | 83.91    |          |
    | - input 224, JFT | 208.2M | 44.66G  | 224   | 84.82    |          |
    | - input 448, JFT | 208.2M | 178.54G | 448   | 86.78    |          |
    | MLPMixerH14, JFT | 432.3M | 121.22G | 224   | 86.32    |          |
    | - input 448, JFT | 432.3M | 484.73G | 448   | 87.94    |          |
## ResMLP
  - [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)
  - [Github facebookresearch/deit](https://github.com/facebookresearch/deit)
  - **Models** reloaded `imagenet` weights are the `distilled` version from official.
    | Model         | Params | FLOPs   | Input | Top1 Acc | Download |
    | ------------- | ------ | ------- | ----- | -------- | -------- |
    | ResMLP12      | 15M    | 3.02G   | 224   | 77.8     | [resmlp12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp12_imagenet.h5) |
    | ResMLP24      | 30M    | 5.98G   | 224   | 80.8     | [resmlp24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp24_imagenet.h5) |
    | ResMLP36      | 116M   | 8.94G   | 224   | 81.1     | [resmlp36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp36_imagenet.h5) |
    | ResMLP_B24    | 129M   | 100.39G | 224   | 83.6     | [resmlp_b24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet.h5) |
    | - imagenet22k | 129M   | 100.39G | 224   | 84.4     | [resmlp_b24_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet22k.h5) |

  - Parameter `pretrained` is added in value `[None, "imagenet", "imagenet22k"]`, where `imagenet22k` means pre-trained on `imagenet21k` and fine-tuned on `imagenet`. Default is `imagenet`.
## GMLP
  - [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).
  - Model weights reloaded from [Github timm/models/mlp_mixer](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py).
  - **Models**
    | Model      | Params | FLOPs  | Input | Top1 Acc | Download |
    | ---------- | ------ | ------ | ----- | -------- | -------- |
    | GMLPTiny16 | 6M     | 1.35G  | 224   | 72.3     |          |
    | GMLPS16    | 20M    | 4.44G  | 224   | 79.6     | [gmlp_s16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/gmlp_s16_imagenet.h5) |
    | GMLPB16    | 73M    | 15.82G | 224   | 81.6     |          |

  - Parameter `pretrained` is added in value `[None, "imagenet"]`. Default is `imagenet`.
## WaveMLP
  - [PDF 2111.12294 An Image Patch is a Wave: Quantum Inspired Vision MLP](https://arxiv.org/pdf/2111.12294.pdf)
  - Model weights reloaded from [Github huawei-noah/wavemlp_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch).
  - **Models**
    | Model     | Params | FLOPs  | Input | Top1 Acc | Download |
    | --------- | ------ | ------ | ----- | -------- | -------- |
    | WaveMLP_T | 17M    | 2.47G  | 224   | 80.9     | [wavemlp_t_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_t_imagenet.h5) |
    | WaveMLP_S | 30M    | 4.55G  | 224   | 82.9     | [wavemlp_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_s_imagenet.h5) |
    | WaveMLP_M | 44M    | 7.92G  | 224   | 83.3     | [wavemlp_m_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_m_imagenet.h5) |
    | WaveMLP_B | 63M    | 10.26G | 224   | 83.6     |          |

  - **Dynamic input shape**
    ```py
    from skimage.data import chelsea
    from keras_cv_attention_models import wave_mlp
    mm = wave_mlp.WaveMLP_T(input_shape=(None, None, 3))
    pred = mm(mm.preprocess_input(chelsea(), input_shape=[320, 320, 3]))
    print(mm.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.4864809), ('n02123159', 'tiger_cat', 0.14551573), ...]
    ```
  - **Verification with PyTorch version**
    ```py
    inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

    """ PyTorch WaveMLP_T """
    sys.path.append("../CV-Backbones")
    from wavemlp_pytorch.models import wavemlp as torch_wavemlp
    import torch
    torch_model = torch_wavemlp.WaveMLP_T()
    ww = torch.load('WaveMLP_T.pth.tar', map_location=torch.device('cpu'))
    ww = {kk: vv for kk, vv in ww.items() if not kk.endswith("total_ops") and not kk.endswith("total_params")}
    torch_model.load_state_dict(ww)
    _ = torch_model.eval()
    torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

    """ Keras WaveMLP_T """
    from keras_cv_attention_models import wave_mlp
    mm = wave_mlp.WaveMLP_T(pretrained="imagenet", classifier_activation=None)
    keras_out = mm(inputs).numpy()

    """ Verification """
    print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
    # np.allclose(torch_out, keras_out, atol=1e-5) = True
    ```
***
