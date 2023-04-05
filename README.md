# Keras_cv_attention_models
- **coco_train_script.py is under testing. Still struggling for this...**
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [General Usage](#general-usage)
  - [Basic](#basic)
  - [Layers](#layers)
  - [Model surgery](#model-surgery)
  - [ImageNet training and evaluating](#imagenet-training-and-evaluating)
  - [COCO training and evaluating](#coco-training-and-evaluating)
  - [Visualizing](#visualizing)
  - [TFLite Conversion](#tflite-conversion)
  - [Using PyTorch as backend](#using-pytorch-as-backend)
- [Recognition Models](#recognition-models)
  - [AotNet](#aotnet)
  - [BEiT](#beit)
  - [BEiTV2](#beitv2)
  - [BotNet](#botnet)
  - [CAFormer](#caformer)
  - [CMT](#cmt)
  - [CoaT](#coat)
  - [CoAtNet](#coatnet)
  - [ConvNeXt](#convnext)
  - [ConvNeXtV2](#convnextv2)
  - [CoTNet](#cotnet)
  - [DaViT](#davit)
  - [EdgeNeXt](#edgenext)
  - [EfficientFormer](#efficientformer)
  - [EfficientFormerV2](#efficientformerv2)
  - [EfficientNet](#efficientnet)
  - [EfficientNetV2](#efficientnetv2)
  - [EVA](#eva)
  - [FasterNet](#fasternet)
  - [FBNetV3](#fbnetv3)
  - [FlexiViT](#flexivit)
  - [GCViT](#gcvit)
  - [GhostNet](#ghostnet)
  - [GhostNetV2](#ghostnetv2)
  - [GMLP](#gmlp)
  - [GPViT](#gpvit)
  - [HaloNet](#halonet)
  - [HorNet](#hornet)
  - [IFormer](#iformer)
  - [InceptionNeXt](#inceptionnext)
  - [LCNet](#lcnet)
  - [LeViT](#levit)
  - [MaxViT](#maxvit)
  - [MLP mixer](#mlp-mixer)
  - [MobileNetV3](#mobilenetv3)
  - [MobileViT](#mobilevit)
  - [MobileViT_V2](#mobilevit_v2)
  - [MogaNet](#moganet)
  - [NAT](#nat)
  - [NFNets](#nfnets)
  - [PVT_V2](#pvt_v2)
  - [RegNetY](#regnety)
  - [RegNetZ](#regnetz)
  - [ResMLP](#resmlp)
  - [ResNeSt](#resnest)
  - [ResNetD](#resnetd)
  - [ResNetQ](#resnetq)
  - [ResNeXt](#resnext)
  - [SwinTransformerV2](#swintransformerv2)
  - [TinyNet](#tinynet)
  - [TinyViT](#tinyvit)
  - [UniFormer](#uniformer)
  - [VOLO](#volo)
  - [WaveMLP](#wavemlp)
- [Detection Models](#detection-models)
  - [EfficientDet](#efficientdet)
  - [YOLOR](#yolor)
  - [YOLOV7](#yolov7)
  - [YOLOX](#yolox)
- [Licenses](#licenses)
- [Citing](#citing)

<!-- /TOC -->
***

# [Roadmap and todo list](https://github.com/leondgarse/keras_cv_attention_models/wiki/Roadmap)
***

# General Usage
## Basic
  - **Currently recommended TF version is `tensorflow==2.10.0`. Expecially for training or TFLite conversion**.
  - **Default import** will not specific these while using them in READMEs.
    ```py
    import os
    import sys
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow import keras
    ```
  - Install as pip package. `kecam` is a short alias name of this package. **Note**: the pip package `kecam` doesn't set any requirement, make sure either Tensorflow or PyTorch installed before hand. For PyTorch backend usage, refer [Keras PyTorch Backend](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/pytorch_backend).
    ```sh
    pip install -U keras-cv-attention-models
    # Or
    pip install -U kecam
    # Or
    pip install -U git+https://github.com/leondgarse/keras_cv_attention_models
    ```
    Refer to each sub directory for detail usage.
  - **Basic model prediction**
    ```py
    from keras_cv_attention_models import volo
    mm = volo.VOLO_d1(pretrained="imagenet")

    """ Run predict """
    import tensorflow as tf
    from tensorflow import keras
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
    Or just use model preset `preprocess_input` and `decode_predictions`
    ```py
    from keras_cv_attention_models import coatnet
    from skimage.data import chelsea
    mm = coatnet.CoAtNet0()
    preds = mm(mm.preprocess_input(chelsea()))
    print(mm.decode_predictions(preds))
    # [[('n02124075', 'Egyptian_cat', 0.9653769), ('n02123159', 'tiger_cat', 0.018427467), ...]
    ```
  - **`num_classes={custom output classes}`** others than `1000` or `0` will just skip loading the header Dense layer weights. As `model.load_weights(weight_file, by_name=True, skip_mismatch=True)` is used for loading weights.
    ```py
    from keras_cv_attention_models import swin_transformer_v2

    mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(num_classes=64)
    # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_window8_256_imagenet.h5
    # WARNING:tensorflow:Skipping loading weights for layer #601 (named predictions) due to mismatch in shape for weight predictions/kernel:0. Weight expects shape (768, 64). Received saved weight with shape (768, 1000)
    # WARNING:tensorflow:Skipping loading weights for layer #601 (named predictions) due to mismatch in shape for weight predictions/bias:0. Weight expects shape (64,). Received saved weight with shape (1000,)
    ```
  - **`num_classes=0`** set for excluding model top `GlobalAveragePooling2D + Dense` layers.
    ```py
    from keras_cv_attention_models import resnest
    mm = resnest.ResNest50(num_classes=0)
    print(mm.output_shape)
    # (None, 7, 7, 2048)
    ```
  - **Reload own model weights by set `pretrained="xxx.h5"`**. Better than calling `model.load_weights` directly, if reloading model with different `input_shape` and with weights shape not matching.
    ```py
    import os
    from keras_cv_attention_models import coatnet
    pretrained = os.path.expanduser('~/.keras/models/coatnet0_224_imagenet.h5')
    mm = coatnet.CoAtNet1(input_shape=(384, 384, 3), pretrained=pretrained)  # No sense, just showing usage
    ```
  - **Alias name `kecam`** can be used instead of `keras_cv_attention_models`. It's `__init__.py` only with `from keras_cv_attention_models import *`.
    ```py
    import kecam
    mm = kecam.yolor.YOLOR_CSP()
    imm = kecam.test_images.dog_cat()
    preds = mm(mm.preprocess_input(imm))
    bboxs, lables, confidences = mm.decode_predictions(preds)[0]
    kecam.coco.show_image_with_bboxes(imm, bboxs, lables, confidences)
    ```
  - **Calculate flops** method from [TF 2.0 Feature: Flops calculation #32809](https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-849439287).
    ```py
    from keras_cv_attention_models import coatnet, resnest, model_surgery

    model_surgery.get_flops(coatnet.CoAtNet0())
    # >>>> FLOPs: 4,221,908,559, GFLOPs: 4.2219G
    model_surgery.get_flops(resnest.ResNest50())
    # >>>> FLOPs: 5,378,399,992, GFLOPs: 5.3784G
    ```
  - **`tensorflow_addons`** is not imported by default. While reloading model depending on `GroupNormalization` like `MobileViTV2` from `h5` directly, needs to import `tensorflow_addons` manually first.
    ```py
    import tensorflow_addons as tfa

    model_path = os.path.expanduser('~/.keras/models/mobilevit_v2_050_256_imagenet.h5')
    mm = keras.models.load_model(model_path)
    ```
  - **Code format** is using `line-length=160`:
    ```sh
    find ./* -name "*.py" | grep -v __init__ | grep -v setup.py | xargs -I {} black -l 160 {}
    ```
## Layers
  - [attention_layers](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/attention_layers) is `__init__.py` only, which imports core layers defined in model architectures. Like `RelativePositionalEmbedding` from `botnet`, `outlook_attention` from `volo`, and many other `Positional Embedding Layers` / `Attention Blocks`.
  ```py
  from keras_cv_attention_models import attention_layers
  aa = attention_layers.RelativePositionalEmbedding()
  print(f"{aa(tf.ones([1, 4, 14, 16, 256])).shape = }")
  # aa(tf.ones([1, 4, 14, 16, 256])).shape = TensorShape([1, 4, 14, 16, 14, 16])
  ```
## Model surgery
  - [model_surgery](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/model_surgery) including functions used to change model parameters after built.
  ```py
  from keras_cv_attention_models import model_surgery
  mm = keras.applications.ResNet50()  # Trainable params: 25,583,592

  # Replace all ReLU with PReLU. Trainable params: 25,606,312
  mm = model_surgery.replace_ReLU(mm, target_activation='PReLU')

  # Fuse conv and batch_norm layers. Trainable params: 25,553,192
  mm = model_surgery.convert_to_fused_conv_bn_model(mm)
  ```
## ImageNet training and evaluating
  - [ImageNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/imagenet) contains more detail usage and some comparing results.
  - [Init Imagenet dataset using tensorflow_datasets #9](https://github.com/leondgarse/keras_cv_attention_models/discussions/9).
  - For custom dataset, `custom_dataset_script.py` can be used creating a `json` format file, which can be used as `--data_name xxx.json` for training, detail usage can be found in [Custom recognition dataset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussion-3971513).
  - Another method creating custom dataset is using `tfds.load`, refer [Writing custom datasets](https://www.tensorflow.org/datasets/add_dataset) and [Creating private tensorflow_datasets from tfds #48](https://github.com/leondgarse/keras_cv_attention_models/discussions/48) by @Medicmind.
  - Running an AWS Sagemaker estimator job using `keras_cv_attention_models` can be found in [AWS Sagemaker script example](https://github.com/leondgarse/keras_cv_attention_models/discussions/107) by @Medicmind.
  - `aotnet.AotNet50` default parameters set is a typical `ResNet50` architecture with `Conv2D use_bias=False` and `padding` like `PyTorch`.
  - Default parameters for `train_script.py` is like `A3` configuration from [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476.pdf) with `batch_size=256, input_shape=(160, 160)`.
    ```sh
    # `antialias` is default enabled for resize, can be turned off be set `--disable_antialias`.
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 -s aotnet50
    ```
    ```sh
    # Evaluation using input_shape (224, 224).
    # `antialias` usage should be same with training.
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m aotnet50_epoch_103_val_acc_0.7674.h5 -i 224 --central_crop 0.95
    # >>>> Accuracy top1: 0.78466 top5: 0.94088
    ```
    ![aotnet50_imagenet](https://user-images.githubusercontent.com/5744524/163795114-b2441e5d-94d5-4310-826a-958426f1343e.png)
  - **Restore from break point** by setting `--restore_path` and `--initial_epoch`, and keep other parameters same. `restore_path` is higher priority than `model` and `additional_model_kwargs`, also restore `optimizer` and `loss`. `initial_epoch` is mainly for learning rate scheduler. If not sure where it stopped, check `checkpoints/{save_name}_hist.json`.
    ```py
    import json
    with open("checkpoints/aotnet50_hist.json", "r") as ff:
        aa = json.load(ff)
    len(aa['lr'])
    # 41 ==> 41 epochs are finished, initial_epoch is 41 then, restart from epoch 42
    ```
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 -r checkpoints/aotnet50_latest.h5 -I 41
    # >>>> Restore model from: checkpoints/aotnet50_latest.h5
    # Epoch 42/105
    ```
  - **`eval_script.py`** is used for evaluating model accuracy. [EfficientNetV2 self tested imagenet accuracy #19](https://github.com/leondgarse/keras_cv_attention_models/discussions/19) just showing how different parameters affecting model accuracy.
    ```sh
    # evaluating pretrained builtin model
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m regnet.RegNetZD8
    # evaluating pretrained timm model
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m timm.models.resmlp_12_224 --input_shape 224

    # evaluating specific h5 model
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m checkpoints/xxx.h5
    # evaluating specific tflite model
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m xxx.tflite
    ```
  - **Progressive training** refer to [PDF 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf). AotNet50 A3 progressive input shapes `96 128 160`:
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./progressive_train_script.py \
    --progressive_epochs 33 66 -1 \
    --progressive_input_shapes 96 128 160 \
    --progressive_magnitudes 2 4 6 \
    -s aotnet50_progressive_3_lr_steps_100 --seed 0
    ```
    ![aotnet50_progressive_160](https://user-images.githubusercontent.com/5744524/151286851-221ff8eb-9fe9-4685-aa60-4a3ba98c654e.png)
  - Transfer learning with `freeze_backbone` or `freeze_norm_layers`: [EfficientNetV2B0 transfer learning on cifar10 testing freezing backbone #55](https://github.com/leondgarse/keras_cv_attention_models/discussions/55).
  - [Token label train test on CIFAR10 #57](https://github.com/leondgarse/keras_cv_attention_models/discussions/57). **Currently not working as well as expected**. `Token label` is implementation of [Github zihangJiang/TokenLabeling](https://github.com/zihangJiang/TokenLabeling), paper [PDF 2104.10858 All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/pdf/2104.10858.pdf).
## COCO training and evaluating
  - **Currently still under testing**.
  - [COCO](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coco) contains more detail usage.
  - `custom_dataset_script.py` can be used creating a `json` format file, which can be used as `--data_name xxx.json` for training, detail usage can be found in [Custom detection dataset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussioncomment-2460664).
  - Default parameters for `coco_train_script.py` is `EfficientDetD0` with `input_shape=(256, 256, 3), batch_size=64, mosaic_mix_prob=0.5, freeze_backbone_epochs=32, total_epochs=105`. Technically, it's any `pyramid structure backbone` + `EfficientDet / YOLOX header / YOLOR header` + `anchor_free / yolor / efficientdet anchors` combination supported.
  - Currently 3 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor"]`. Default `None` for `det_header` presets.

    | anchors_mode | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------ | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free  | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor        | True              | 3           | None         | presets       | None       | offset=0.5      |

    ```sh
    # Default EfficientDetD0
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py
    # Default EfficientDetD0 using input_shape 512, optimizer adamw, freezing backbone 16 epochs, total 50 + 5 epochs
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py -i 512 -p adamw --freeze_backbone_epochs 16 --lr_decay_steps 50

    # EfficientNetV2B0 backbone + EfficientDetD0 detection header
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone efficientnet.EfficientNetV2B0 --det_header efficientdet.EfficientDetD0
    # ResNest50 backbone + EfficientDetD0 header using yolox like anchor_free anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone resnest.ResNest50 --anchors_mode anchor_free
    # UniformerSmall32 backbone + EfficientDetD0 header using yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone uniformer.UniformerSmall32 --anchors_mode yolor

    # Typical YOLOXS with anchor_free anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS --freeze_backbone_epochs 0
    # YOLOXS with efficientdet anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS --anchors_mode efficientdet --freeze_backbone_epochs 0
    # CoAtNet0 backbone + YOLOX header with yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone coatnet.CoAtNet0 --det_header yolox.YOLOX --anchors_mode yolor

    # Typical YOLOR_P6 with yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolor.YOLOR_P6 --freeze_backbone_epochs 0
    # YOLOR_P6 with anchor_free anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolor.YOLOR_P6 --anchors_mode anchor_free  --freeze_backbone_epochs 0
    # ConvNeXtTiny backbone + YOLOR header with efficientdet anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone convnext.ConvNeXtTiny --det_header yolor.YOLOR --anchors_mode yolor
    ```
    **Note: COCO training still under testing, may change parameters and default behaviors. Take the risk if would like help developing.**
  - **`coco_eval_script.py`** is used for evaluating model AP / AR on COCO validation set. It has a dependency `pip install pycocotools` which is not in package requirements. More usage can be found in [COCO Evaluation](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coco#evaluation).
    ```sh
    # EfficientDetD0 using resize method bilinear w/o antialias
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m efficientdet.EfficientDetD0 --resize_method bilinear --disable_antialias
    # >>>> [COCOEvalCallback] input_shape: (512, 512), pyramid_levels: [3, 7], anchors_mode: efficientdet

    # YOLOX using BGR input format
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolox.YOLOXTiny --use_bgr_input --nms_method hard --nms_iou_or_sigma 0.65
    # >>>> [COCOEvalCallback] input_shape: (416, 416), pyramid_levels: [3, 5], anchors_mode: anchor_free

    # YOLOR / YOLOV7 using letterbox_pad and other tricks.
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolor.YOLOR_CSP --nms_method hard --nms_iou_or_sigma 0.65 \
    --nms_max_output_size 300 --nms_topk -1 --letterbox_pad 64 --input_shape 704
    # >>>> [COCOEvalCallback] input_shape: (704, 704), pyramid_levels: [3, 5], anchors_mode: yolor

    # Specify h5 model
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m checkpoints/yoloxtiny_yolor_anchor.h5
    # >>>> [COCOEvalCallback] input_shape: (416, 416), pyramid_levels: [3, 5], anchors_mode: yolor
    ```
## Visualizing
  - [Visualizing](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/visualizing) is for visualizing convnet filters or attention map scores.
  - **make_and_apply_gradcam_heatmap** is for Grad-CAM class activation visualization.
    ```py
    from keras_cv_attention_models import visualizing, test_images, resnest
    mm = resnest.ResNest50()
    img = test_images.dog()
    superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, img, layer_name="auto")
    ```
    ![](https://user-images.githubusercontent.com/5744524/148199374-4944800e-a1fb-4df2-b9ba-43ce3dde88f2.png)
  - **plot_attention_score_maps** is model attention score maps visualization.
    ```py
    from keras_cv_attention_models import visualizing, test_images, botnet
    img = test_images.dog()
    _ = visualizing.plot_attention_score_maps(botnet.BotNetSE33T(), img)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209511-f5194d73-9e4c-457e-a763-45a4025f452b.png)
## TFLite Conversion
  - Currently `TFLite` not supporting `Conv2D with groups>1` / `gelu` / `tf.image.extract_patches` / `tf.transpose with len(perm) > 4`. Some operations could be supported in `tf-nightly` version. May try if encountering issue. More discussion can be found [Converting a trained keras CV attention model to TFLite #17](https://github.com/leondgarse/keras_cv_attention_models/discussions/17). Some speed testing results can be found [How to speed up inference on a quantized model #44](https://github.com/leondgarse/keras_cv_attention_models/discussions/44#discussioncomment-2348910).
  - `tf.nn.gelu(inputs, approximate=True)` activation works for TFLite. Define model with `activation="gelu/approximate"` or `activation="gelu/app"` will set `approximate=True` for `gelu`. **Should better decide before training, or there may be accuracy loss**.
  - Not supporting `VOLO` / `HaloNet` models converting, cause they need a longer `tf.transpose` `perm`.
  - **model_surgery.convert_groups_conv2d_2_split_conv2d** converts model `Conv2D with groups>1` layers to `SplitConv` using `split -> conv -> concat`:
    ```py
    from keras_cv_attention_models import regnet, model_surgery
    from keras_cv_attention_models.imagenet import eval_func

    bb = regnet.RegNetZD32()
    mm = model_surgery.convert_groups_conv2d_2_split_conv2d(bb)  # converts all `Conv2D` using `groups` to `SplitConv2D`
    test_inputs = np.random.uniform(size=[1, *mm.input_shape[1:]])
    print(np.allclose(mm(test_inputs), bb(test_inputs)))
    # True

    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    open(mm.name + ".tflite", "wb").write(converter.convert())
    print(np.allclose(mm(test_inputs), eval_func.TFLiteModelInterf(mm.name + '.tflite')(test_inputs), atol=1e-7))
    # True
    ```
  - **model_surgery.convert_gelu_and_extract_patches_for_tflite** converts model `gelu` activation to `gelu approximate=True`, and `tf.image.extract_patches` to a `Conv2D` version:
    ```py
    from keras_cv_attention_models import cotnet, model_surgery
    from keras_cv_attention_models.imagenet import eval_func

    mm = cotnet.CotNetSE50D()
    mm = model_surgery.convert_groups_conv2d_2_split_conv2d(mm)
    mm = model_surgery.convert_gelu_and_extract_patches_for_tflite(mm)
    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    open(mm.name + ".tflite", "wb").write(converter.convert())
    test_inputs = np.random.uniform(size=[1, *mm.input_shape[1:]])
    print(np.allclose(mm(test_inputs), eval_func.TFLiteModelInterf(mm.name + '.tflite')(test_inputs), atol=1e-7))
    # True
    ```
  - **model_surgery.prepare_for_tflite** is just a combination of above 2 functions:
    ```py
    from keras_cv_attention_models import beit, model_surgery

    mm = beit.BeitBasePatch16()
    mm = model_surgery.prepare_for_tflite(mm)
    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    open(mm.name + ".tflite", "wb").write(converter.convert())
    ```
  - **Detection models** including `efficinetdet` / `yolox` / `yolor`, model can be converted a TFLite format directly. If need [DecodePredictions](https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/coco/eval_func.py#L8) also included in TFLite model, need to set `use_static_output=True` for `DecodePredictions`, as TFLite requires a more static output shape. Model output shape will be fixed as `[batch, max_output_size, 6]`. The last dimension `6` means `[bbox_top, bbox_left, bbox_bottom, bbox_right, label_index, confidence]`, and those valid ones are where `confidence > 0`.
    ```py
    """ Init model """
    from keras_cv_attention_models import efficientdet
    model = efficientdet.EfficientDetD0(pretrained="coco")

    """ Create a model with DecodePredictions using `use_static_output=True` """
    model.decode_predictions.use_static_output = True
    # parameters like score_threshold / iou_or_sigma can be set another value if needed.
    nn = model.decode_predictions(model.outputs[0], score_threshold=0.5)
    bb = keras.models.Model(model.inputs[0], nn)

    """ Convert TFLite """
    converter = tf.lite.TFLiteConverter.from_keras_model(bb)
    open(bb.name + ".tflite", "wb").write(converter.convert())

    """ Inference test """
    from keras_cv_attention_models.imagenet import eval_func
    from keras_cv_attention_models import test_images

    dd = eval_func.TFLiteModelInterf(bb.name + ".tflite")
    imm = test_images.cat()
    inputs = tf.expand_dims(tf.image.resize(imm, dd.input_shape[1:-1]), 0)
    inputs = keras.applications.imagenet_utils.preprocess_input(inputs, mode='torch')
    preds = dd(inputs)[0]
    print(f"{preds.shape = }")
    # preds.shape = (100, 6)

    pred = preds[preds[:, -1] > 0]
    bboxes, labels, confidences = pred[:, :4], pred[:, 4], pred[:, -1]
    print(f"{bboxes = }, {labels = }, {confidences = }")
    # bboxes = array([[0.22825494, 0.47238672, 0.816262  , 0.8700745 ]], dtype=float32),
    # labels = array([16.], dtype=float32),
    # confidences = array([0.8309707], dtype=float32)

    """ Show result """
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxes, labels, confidences, num_classes=90)
    ```
## Using PyTorch as backend
  - **Experimental** [Keras PyTorch Backend](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/pytorch_backend).
  - **Set os environment `export KECAM_BACKEND='torch'` to enable this PyTorch backend.**
  - Currently supports most recognition and detection models except cotnet / halonet / hornet / nat / nfnets / volo. For detection models, still using `tf.image.non_max_suppression_with_scores` while running prediction.
  - **Basic model build and prediction**.
    - Will load same `h5` weights as TF one if available.
    - Note: `input_shape` will auto fit image data format. Given `input_shape=(224, 224, 3)` or `input_shape=(3, 224, 224)`, will both set to `(3, 224, 224)` if `channels_first`.
    - Note: model is default set to `eval` mode.
    ```py
    from keras_cv_attention_models import res_mlp
    mm = res_mlp.ResMLP12()
    # >>>> Load pretrained from: ~/.keras/models/resmlp12_imagenet.h5
    print(f"{mm.input_shape = }")
    # mm.input_shape = [None, 3, 224, 224]

    import torch
    print(f"{isinstance(mm, torch.nn.Module) = }")
    # isinstance(mm, torch.nn.Module) = True

    # Run prediction
    from skimage.data import chelsea # Chelsea the cat
    print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
    # [('n02124075', 'Egyptian_cat', 0.86188155), ('n02123159', 'tiger_cat', 0.05125639), ...]
    ```
  - **Export typical PyTorch onnx / pth**.
    ```py
    import torch
    torch.onnx.export(mm, torch.randn(1, 3, *mm.input_shape[2:]), mm.name + ".onnx")

    # Or by export_onnx
    mm.export_onnx()
    # Exported onnx: resmlp12.onnx

    mm.export_pth()
    # Exported pth: resmlp12.pth
    ```
  - **Save weights as h5**. This `h5` can also be loaded in typical TF backend model. Currently it's only weights without model structure supported.
    ```py
    mm.save_weights("foo.h5")
    ```
***

# Recognition Models
## AotNet
  - [Keras AotNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/aotnet) is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer. Works like `byoanet` / `byobnet` from `timm`.
  - Default parameters set is a typical `ResNet` architecture with `Conv2D use_bias=False` and `padding` like `PyTorch`.
  ```py
  from keras_cv_attention_models import aotnet
  # Mixing se and outlook and halo and mhsa and cot_attention, 21M parameters.
  # 50 is just a picked number that larger than the relative `num_block`.
  attn_types = [None, "outlook", ["bot", "halo"] * 50, "cot"],
  se_ratio = [0.25, 0, 0, 0],
  model = aotnet.AotNet50V2(attn_types=attn_types, se_ratio=se_ratio, stem_type="deep", strides=1)
  model.summary()
  ```
## BEiT
  - [Keras BEiT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/beit) includes models from [PDF 2106.08254 BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).

  | Model                 | Params  | FLOPs   | Input | Top1 Acc | Download                         |
  | --------------------- | ------- | ------- | ----- | -------- | -------------------------------- |
  | BeitBasePatch16, 21k  | 86.53M  | 17.61G  | 224   | 85.240   | [beit_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_224_imagenet21k-ft1k.h5)  |
  |                       | 86.74M  | 55.70G  | 384   | 86.808   | [beit_base_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_384_imagenet21k-ft1k.h5)  |
  | BeitLargePatch16, 21k | 304.43M | 61.68G  | 224   | 87.476   | [beit_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_224_imagenet21k-ft1k.h5) |
  |                       | 305.00M | 191.65G | 384   | 88.382   | [beit_large_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_384_imagenet21k-ft1k.h5) |
  |                       | 305.67M | 363.46G | 512   | 88.584   | [beit_large_patch16_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_512_imagenet21k-ft1k.h5) |
## BEiTV2
  - [Keras BEiT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/beit) includes models from BeitV2 Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).

  | Model              | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------- | ------ | ----- | -------- | -------- |
  | BeitV2BasePatch16  | 86.53M  | 17.61G | 224   | 85.5     |          |
  | - imagenet21k-ft1k | 86.53M  | 17.61G | 224   | 86.5     | [beit_v2_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_base_patch16_224_imagenet21k-ft1k.h5)  |
  | BeitV2BasePatch16  | 304.43M | 61.68G | 224   | 87.3     |          |
  | - imagenet21k-ft1k | 304.43M | 61.68G | 224   | 88.4     | [beit_v2_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_large_patch16_224_imagenet21k-ft1k.h5)  |
## BotNet
  - [Keras BotNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/botnet) is for [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | BotNet50      | 21M    | 5.42G  | 224   |          |          |
  | BotNet101     | 41M    | 9.13G  | 224   |          |          |
  | BotNet152     | 56M    | 12.84G | 224   |          |          |
  | BotNet26T     | 12.5M  | 3.30G  | 256   | 79.246   | [botnet26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet26t_256_imagenet.h5) |
  | BotNextECA26T | 10.59M | 2.45G  | 256   | 79.270   | [botnext_eca26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnext_eca26t_256_imagenet.h5) |
  | BotNetSE33T   | 13.7M  | 3.89G  | 256   | 81.2     | [botnet_se33t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet_se33t_256_imagenet.h5) |
## CAFormer
  - [Keras CAFormer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/caformer) is for [PDF 2210.13452 MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452.pdf). `CAFormer` is using 2 transformer stacks, while `ConvFormer` is all conv blocks.

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | CAFormerS18        | 26M    | 4.1G  | 224   | 83.6     | [caformer_s18_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet.h5) |
  | - imagenet21k-ft1k | 26M    | 4.1G  | 224   | 84.1     | [caformer_s18_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet21k-ft1k.h5) |
  |                    | 26M    | 13.4G | 384   | 85.0     | [caformer_s18_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet.h5) |
  | - imagenet21k-ft1k | 26M    | 13.4G | 384   | 85.4     | [caformer_s18_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet21k-ft1k.h5) |
  | CAFormerS36        | 39M    | 8.0G  | 224   | 84.5     | [caformer_s36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 39M    | 8.0G  | 224   | 85.8     | [caformer_s36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet21k-ft1k.h5) |
  |                    | 39M    | 26.0G | 384   | 85.7     | [caformer_s36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 39M    | 26.0G | 384   | 86.9     | [caformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet21k-ft1k.h5) |
  | CAFormerM36        | 56M    | 13.2G | 224   | 85.2     | [caformer_m36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 56M    | 13.2G | 224   | 86.6     | [caformer_m36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet21k-ft1k.h5) |
  |                    | 56M    | 42.0G | 384   | 86.2     | [caformer_m36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 56M    | 42.0G | 384   | 87.5     | [caformer_m36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet21k-ft1k.h5) |
  | CAFormerB36        | 99M    | 23.2G | 224   | 85.5     | [caformer_b36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 99M    | 23.2G | 224   | 87.4     | [caformer_b36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet21k-ft1k.h5) |
  |                    | 99M    | 72.2G | 384   | 86.4     | [caformer_b36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 99M    | 72.2G | 384   | 88.1     | [caformer_b36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet21k-ft1k.h5) |

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | ConvFormerS18      | 27M    | 3.9G  | 224   | 83.0     | [convformer_s18_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet.h5) |
  | - imagenet21k-ft1k | 27M    | 3.9G  | 224   | 83.7     | [convformer_s18_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet21k-ft1k.h5) |
  |                    | 27M    | 11.6G | 384   | 84.4     | [convformer_s18_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_384_imagenet.h5) |
  | - imagenet21k-ft1k | 27M    | 11.6G | 384   | 85.0     | [convformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) |
  | ConvFormerS36      | 40M    | 7.6G  | 224   | 84.1     | [convformer_s36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 40M    | 7.6G  | 224   | 85.4     | [convformer_s36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet21k-ft1k.h5) |
  |                    | 40M    | 22.4G | 384   | 85.4     | [convformer_s36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 40M    | 22.4G | 384   | 86.4     | [convformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) |
  | ConvFormerM36      | 57M    | 12.8G | 224   | 84.5     | [convformer_m36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 57M    | 12.8G | 224   | 86.1     | [convformer_m36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet21k-ft1k.h5) |
  |                    | 57M    | 37.7G | 384   | 85.6     | [convformer_m36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 57M    | 37.7G | 384   | 86.9     | [convformer_m36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet21k-ft1k.h5) |
  | ConvFormerB36      | 100M   | 22.6G | 224   | 84.8     | [convformer_b36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 100M   | 22.6G | 224   | 87.0     | [convformer_b36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet21k-ft1k.h5) |
  |                    | 100M   | 66.5G | 384   | 85.7     | [convformer_b36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 100M   | 66.5G | 384   | 87.6     | [convformer_b36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet21k-ft1k.h5) |
## CMT
  - [Keras CMT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/cmt) is for [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf).

  | Model                              | Params | FLOPs | Input | Top1 Acc | Download |
  | ---------------------------------- | ------ | ----- | ----- | -------- | -------- |
  | CMTTiny, (Self trained 105 epochs) | 9.5M   | 0.65G | 160   | 77.4     |          |
  | - 305 epochs                       | 9.5M   | 0.65G | 160   | 78.94    | [cmt_tiny_160_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_160_imagenet.h5) |
  | - fine-tuned 224 (69 epochs)       | 9.5M   | 1.32G | 224   | 80.73    | [cmt_tiny_224_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_224_imagenet.h5) |
  | CMTTiny_torch, 1000 epochs         | 9.5M   | 0.65G | 160   | 79.2     | [cmt_tiny_torch_160](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_torch_160_imagenet.h5) |
  | CMTXS_torch                        | 15.2M  | 1.58G | 192   | 81.8     | [cmt_xs_torch_192](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_xs_torch_192_imagenet.h5) |
  | CMTSmall_torch                     | 25.1M  | 4.09G | 224   | 83.5     | [cmt_small_torch_224](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_small_torch_224_imagenet.h5) |
  | CMTBase_torch                      | 45.7M  | 9.42G | 256   | 84.5     | [cmt_base_torch_256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_base_torch_256_imagenet.h5) |
## CoaT
  - [Keras CoaT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coat) is for [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).

  | Model         | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------- | ------ | ----- | ----- | -------- | -------- |
  | CoaTLiteTiny  | 5.7M   | 1.60G | 224   | 77.5     | [coat_lite_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_tiny_imagenet.h5) |
  | CoaTLiteMini  | 11M    | 2.00G | 224   | 79.1     | [coat_lite_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_mini_imagenet.h5) |
  | CoaTLiteSmall | 20M    | 3.97G | 224   | 81.9     | [coat_lite_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_small_imagenet.h5) |
  | CoaTTiny      | 5.5M   | 4.33G | 224   | 78.3     | [coat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_tiny_imagenet.h5) |
  | CoaTMini      | 10M    | 6.78G | 224   | 81.0     | [coat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_mini_imagenet.h5) |
## CoAtNet
  - [Keras CoAtNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coatnet) is for [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf).

  | Model                               | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------------------------- | ------ | ------ | ----- | -------- | -------- |
  | CoAtNet0 (Self trained 105 epochs)  | 23.3M  | 2.09G  | 160   | 80.48    | [coatnet0_160_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_160_imagenet.h5) |
  | CoAtNet0 (Self trained 305 epochs)  | 23.8M  | 4.22G  | 224   | 82.79    | [coatnet0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_224_imagenet.h5) |
  | CoAtNet0                            | 25M    | 4.2G   | 224   | 81.6     |          |
  | CoAtNet0, Stride-2 DConv2D          | 25M    | 4.6G   | 224   | 82.0     |          |
  | CoAtNet1                            | 42M    | 8.4G   | 224   | 83.3     |          |
  | CoAtNet1, Stride-2 DConv2D          | 42M    | 8.8G   | 224   | 83.5     |          |
  | CoAtNet2                            | 75M    | 15.7G  | 224   | 84.1     |          |
  | CoAtNet2, Stride-2 DConv2D          | 75M    | 16.6G  | 224   | 84.1     |          |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 16.6G  | 224   | 87.1     |          |
  | CoAtNet3                            | 168M   | 34.7G  | 224   | 84.5     |          |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 34.7G  | 224   | 87.6     |          |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 203.1G | 512   | 87.9     |          |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 360.9G | 512   | 88.1     |          |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 360.9G | 512   | 88.56    |          |

  **JFT pre-trained models accuracy**

  | Model                      | Input | Reported Params    | self-defined Params    | Top1 Acc |
  | -------------------------- | ----- | ------------------ | ---------------------- | -------- |
  | CoAtNet3, Stride-2 DConv2D | 384   | 168M, FLOPs 114G   | 160.64M, FLOPs 109.67G | 88.52    |
  | CoAtNet3, Stride-2 DConv2D | 512   | 168M, FLOPs 214G   | 161.24M, FLOPs 205.06G | 88.81    |
  | CoAtNet4                   | 512   | 275M, FLOPs 361G   | 270.69M, FLOPs 359.77G | 89.11    |
  | CoAtNet5                   | 512   | 688M, FLOPs 812G   | 676.23M, FLOPs 807.06G | 89.77    |
  | CoAtNet6                   | 512   | 1.47B, FLOPs 1521G | 1.336B, FLOPs 1470.56G | 90.45    |
  | CoAtNet7                   | 512   | 2.44B, FLOPs 2586G | 2.413B, FLOPs 2537.56G | 90.88    |
## ConvNeXt
  - [Keras ConvNeXt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/convnext) is for [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf).

  | Model               | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------------- | ------ | ------- | ----- | -------- | -------- |
  | ConvNeXtTiny        | 28M    | 4.49G   | 224   | 82.1     | [tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_imagenet.h5) |
  | - ImageNet21k-ft1k  | 28M    | 4.49G   | 224   | 82.9     | [tiny_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 28M    | 13.19G  | 384   | 84.1     | [tiny_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_384_imagenet21k-ft1k.h5) |
  | ConvNeXtSmall       | 50M    | 8.73G   | 224   | 83.1     | [small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_imagenet.h5) |
  | - ImageNet21k-ft1k  | 50M    | 8.73G   | 224   | 84.6     | [small_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 50M    | 25.67G  | 384   | 85.8     | [small_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_384_imagenet21k-ft1k.h5) |
  | ConvNeXtBase        | 89M    | 15.42G  | 224   | 83.8     | [base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet.h5) |
  | ConvNeXtBase        | 89M    | 45.32G  | 384   | 85.1     | [base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet.h5) |
  | - ImageNet21k-ft1k  | 89M    | 15.42G  | 224   | 85.8     | [base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 89M    | 45.32G  | 384   | 86.8     | [base_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet21k-ft1k.h5) |
  | ConvNeXtLarge       | 198M   | 34.46G  | 224   | 84.3     | [large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet.h5) |
  | ConvNeXtLarge       | 198M   | 101.28G | 384   | 85.5     | [large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet.h5) |
  | - ImageNet21k-ft1k  | 198M   | 34.46G  | 224   | 86.6     | [large_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 198M   | 101.28G | 384   | 87.5     | [large_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet21k-ft1k.h5) |
  | ConvNeXtXLarge, 21k | 350M   | 61.06G  | 224   | 87.0     | [xlarge_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_224_imagenet21k-ft1k.h5) |
  | ConvNeXtXLarge, 21k | 350M   | 179.43G | 384   | 87.8     | [xlarge_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_384_imagenet21k-ft1k.h5) |
## ConvNeXtV2
  - [Keras ConvNeXt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/convnext) includes implementation of [PDF 2301.00808 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808.pdf). **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.

  | Model              | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------ | ------ | ----- | -------- | -------- |
  | ConvNeXtV2Atto     | 3.7M   | 0.55G  | 224   | 76.7     | [v2_atto_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_atto_imagenet.h5) |
  | ConvNeXtV2Femto    | 5.2M   | 0.78G  | 224   | 78.5     | [v2_femto_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_femto_imagenet.h5) |
  | ConvNeXtV2Pico     | 9.1M   | 1.37G  | 224   | 80.3     | [v2_pico_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_pico_imagenet.h5) |
  | ConvNeXtV2Nano     | 15.6M  | 2.45G  | 224   | 81.9     | [v2_nano_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_imagenet.h5) |
  | - ImageNet21k-ft1k | 15.6M  | 2.45G  | 224   | 82.1     | [v2_nano_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 15.6M  | 7.21G  | 384   | 83.4     | [v2_nano_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Tiny     | 28.6M  | 4.47G  | 224   | 83.0     | [v2_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_imagenet.h5) |
  | - ImageNet21k-ft1k | 28.6M  | 4.47G  | 224   | 83.9     | [v2_tiny_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 28.6M  | 13.1G  | 384   | 85.1     | [v2_tiny_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Base     | 89M    | 15.4G  | 224   | 84.9     | [v2_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_imagenet.h5) |
  | - ImageNet21k-ft1k | 89M    | 15.4G  | 224   | 86.8     | [v2_base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 89M    | 45.2G  | 384   | 87.7     | [v2_base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Large    | 198M   | 34.4G  | 224   | 85.8     | [v2_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_imagenet.h5) |
  | - ImageNet21k-ft1k | 198M   | 34.4G  | 224   | 87.3     | [v2_large_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 198M   | 101.1G | 384   | 88.2     | [v2_large_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Huge     | 660M   | 115G   | 224   | 86.3     | [v2_huge_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_imagenet.h5) |
  | - ImageNet21k-ft1k | 660M   | 337.9G | 384   | 88.7     | [v2_huge_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_384_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 660M   | 600.8G | 512   | 88.9     | [v2_huge_512_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_512_imagenet21k-ft1k.h5) |
## CoTNet
  - [Keras CoTNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/cotnet) is for [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | Download            |
  | ------------ |:------:| ------ | ----- |:--------:| ------------------- |
  | CotNet50     | 22.2M  | 3.25G  | 224   |   81.3   | [cotnet50_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224_imagenet.h5) |
  | CotNeXt50    | 30.1M  | 4.3G   | 224   |   82.1   |  |
  | CotNetSE50D  | 23.1M  | 4.05G  | 224   |   81.6   | [cotnet_se50d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se50d_224_imagenet.h5) |
  | CotNet101    | 38.3M  | 6.07G  | 224   |   82.8   | [cotnet101_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224_imagenet.h5) |
  | CotNeXt101   | 53.4M  | 8.2G   | 224   |   83.2   |  |
  | CotNetSE101D | 40.9M  | 8.44G  | 224   |   83.2   | [cotnet_se101d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se101d_224_imagenet.h5) |
  | CotNetSE152D | 55.8M  | 12.22G | 224   |   84.0   | [cotnet_se152d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_224_imagenet.h5) |
  | CotNetSE152D | 55.8M  | 24.92G | 320   |   84.6   | [cotnet_se152d_320_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_320_imagenet.h5) |
## DaViT
  - [Keras DaViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/davit) is for [PDF 2204.03645 DaViT: Dual Attention Vision Transformers](https://arxiv.org/pdf/2204.03645.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | DaViT_T       | 28.36M | 4.56G  | 224   | 82.8     | [davit_t_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_t_imagenet.h5) |
  | DaViT_S       | 49.75M | 8.83G  | 224   | 84.2     | [davit_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_s_imagenet.h5) |
  | DaViT_B       | 87.95M | 15.55G | 224   | 84.6     | [davit_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_b_imagenet.h5) |
  | DaViT_L, 21k  | 196.8M | 103.2G | 384   | 87.5     |          |
  | DaViT_H, 1.5B | 348.9M | 327.3G | 512   | 90.2     |          |
  | DaViT_G, 1.5B | 1.406B | 1.022T | 512   | 90.4     |          |
## EdgeNeXt
  - [Keras EdgeNeXt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/edgenext) is for [PDF 2206.10589 EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/pdf/2206.10589.pdf).

  | Model             | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------- | ------ | ------ | ----- | -------- | -------- |
  | EdgeNeXt_XX_Small | 1.33M  | 266M   | 256   | 71.23    | [edgenext_xx_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_xx_small_256_imagenet.h5) |
  | EdgeNeXt_X_Small  | 2.34M  | 547M   | 256   | 74.96    | [edgenext_x_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_x_small_256_imagenet.h5) |
  | EdgeNeXt_Small    | 5.59M  | 1.27G  | 256   | 79.41    | [edgenext_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_imagenet.h5) |
  | - usi             | 5.59M  | 1.27G  | 256   | 81.07    | [edgenext_small_256_usi.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_usi.h5) |
## EfficientFormer
  - [Keras EfficientFormer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/efficientformer) is for [PDF 2206.01191 EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191.pdf).

  | Model                      | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------------------- | ------ | ----- | ----- | -------- | -------- |
  | EfficientFormerL1, distill | 12.3M  | 1.31G | 224   | 79.2     | [l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l1_224_imagenet.h5) |
  | EfficientFormerL3, distill | 31.4M  | 3.95G | 224   | 82.4     | [l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l3_224_imagenet.h5) |
  | EfficientFormerL7, distill | 74.4M  | 9.79G | 224   | 83.3     | [l7_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l7_224_imagenet.h5) |
## EfficientFormerV2
  - [Keras EfficientFormer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/efficientformer) includes implementation of [PDF 2212.08059 Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/pdf/2212.08059.pdf).

  | Model                        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ---------------------------- | ------ | ------ | ----- | -------- | -------- |
  | EfficientFormerV2S0, distill | 3.60M  | 405.2M | 224   | 76.2     | [v2_s0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s0_224_imagenet.h5) |
  | EfficientFormerV2S1, distill | 6.19M  | 665.6M | 224   | 79.7     | [v2_s1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s1_224_imagenet.h5) |
  | EfficientFormerV2S2, distill | 12.7M  | 1.27G  | 224   | 82.0     | [v2_s2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s2_224_imagenet.h5) |
  | EfficientFormerV2L, distill  | 26.3M  | 2.59G  | 224   | 83.5     | [v2_l_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_l_224_imagenet.h5) |
## EfficientNet
  - [Keras EfficientNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/efficientnet) includes implementation of [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf).

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
## EfficientNetV2
  - [Keras EfficientNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/efficientnet) includes implementation of [PDF 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

  | V2 Model                   | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------------------------- | ------ | ------ | ----- | -------- | -------- |
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
## EVA
  - [Keras EVA](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/beit) includes models from [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf).

  | Model                 | Params  | FLOPs    | Input | Top1 Acc | Download |
  | --------------------- | ------- | -------- | ----- | -------- | -------- |
  | EvaLargePatch14, 22k  | 304.14M | 61.65G   | 196   | 88.59    | [eva_large_patch14_196.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_196_imagenet21k-ft1k.h5) |
  |                       | 304.53M | 191.55G  | 336   | 89.20    | [eva_large_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_336_imagenet21k-ft1k.h5) |
  | EvaGiantPatch14, clip | 1012.6M | 267.40G  | 224   | 89.10    | [eva_giant_patch14_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_224_imagenet21k-ft1k.h5) |
  | - m30m                | 1013.0M | 621.45G  | 336   | 89.57    | [eva_giant_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_336_imagenet21k-ft1k.h5) |
  | - m30m                | 1014.4M | 1911.61G | 560   | 89.80    | [eva_giant_patch14_560.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_560_imagenet21k-ft1k.h5) |
## FasterNet
  - [Keras FasterNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/fasternet) includes implementation of [PDF 2303.03667 Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks ](https://arxiv.org/pdf/2303.03667.pdf).

  | Model       | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------- | ------ | ------ | ----- | -------- | -------- |
  | FasterNetT0 | 3.9M   | 0.34G  | 224   | 71.9     | [fasternet_t0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t0_imagenet.h5) |
  | FasterNetT1 | 7.6M   | 0.85G  | 224   | 76.2     | [fasternet_t1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t1_imagenet.h5) |
  | FasterNetT2 | 15.0M  | 1.90G  | 224   | 78.9     | [fasternet_t2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t2_imagenet.h5) |
  | FasterNetS  | 31.1M  | 4.55G  | 224   | 81.3     | [fasternet_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_s_imagenet.h5)   |
  | FasterNetM  | 53.5M  | 8.72G  | 224   | 83.0     | [fasternet_m_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_m_imagenet.h5)   |
  | FasterNetL  | 93.4M  | 15.49G | 224   | 83.5     | [fasternet_l_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_l_imagenet.h5)   |
## FBNetV3
  - [Keras FBNetV3](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilenetv3_family#fbnetv3) includes implementation of [PDF 2006.02049 FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining](https://arxiv.org/pdf/2006.02049.pdf).

  | Model    | Params | FLOPs    | Input | Top1 Acc | Download |
  | -------- | ------ | -------- | ----- | -------- | -------- |
  | FBNetV3B | 5.57M  | 539.82M  | 256   | 79.15    | [fbnetv3_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_b_imagenet.h5) |
  | FBNetV3D | 10.31M | 665.02M  | 256   | 79.68    | [fbnetv3_d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_d_imagenet.h5) |
  | FBNetV3G | 16.62M | 1379.30M | 256   | 82.05    | [fbnetv3_g_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_g_imagenet.h5) |
## FlexiViT
  - [Keras FlexiViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/beit) includes models from [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).

  | Model         | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------- | ------ | ----- | -------- | -------- |
  | FlexiViTSmall | 22.06M  | 5.36G  | 240   | 82.53    | [flexivit_small_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_small_240_imagenet.h5) |
  | FlexiViTBase  | 86.59M  | 20.33G | 240   | 84.66    | [flexivit_base_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_base_240_imagenet.h5) |
  | FlexiViTLarge | 304.47M | 71.09G | 240   | 85.64    | [flexivit_large_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_large_240_imagenet.h5) |
## GCViT
  - [Keras GCViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/gcvit) includes implementation of [PDF 2206.09959 Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf).

  | Model        | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------ | ------ | ----- | ----- | -------- | -------- |
  | GCViT_XXTiny | 12.0M  | 2.15G | 224   | 79.8     | [gcvit_xx_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_xx_tiny_224_imagenet.h5) |
  | GCViT_XTiny  | 20.0M  | 2.96G | 224   | 82.04    | [gcvit_x_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_x_tiny_224_imagenet.h5) |
  | GCViT_Tiny   | 28.2M  | 4.83G | 224   | 83.4     | [gcvit_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_tiny_224_imagenet.h5) |
  | GCViT_Small  | 51.1M  | 8.63G | 224   | 83.95    | [gcvit_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_small_224_imagenet.h5) |
  | GCViT_Base   | 90.3M  | 14.9G | 224   | 84.47    | [gcvit_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_base_224_imagenet.h5) |
## GhostNet
  - [Keras GhostNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/ghostnet) includes implementation of [PDF 1911.11907 GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | GhostNet_050 | 2.59M  | 42.6M  | 224   | 66.88    | [ghostnet_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_050_imagenet.h5) |
  | GhostNet_100 | 5.18M  | 141.7M | 224   | 74.16    | [ghostnet_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_100_imagenet.h5) |
  | GhostNet_130 | 7.36M  | 227.7M | 224   | 75.79    | [ghostnet_130_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_imagenet.h5) |
  | - ssld       | 7.36M  | 227.7M | 224   | 79.38    | [ghostnet_130_ssld.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_ssld.h5) |
## GhostNetV2
  - [Keras GhostNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/ghostnet) includes implementation of [PDF GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf).

  | Model             | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------- | ------ | ------ | ----- | -------- | -------- |
  | GhostNetV2_100    | 6.12M  | 168.5M | 224   | 74.41    | [ghostnetv2_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnetv2_100_imagenet.h5) |
  | GhostNetV2 (1.0x) | 6.12M  | 168.5M | 224   | 75.3     |          |
  | GhostNetV2 (1.3x) | 8.96M  | 271.1M | 224   | 76.9     |          |
  | GhostNetV2 (1.6x) | 12.39M | 400.9M | 224   | 77.8     |          |
## GMLP
  - [Keras GMLP](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mlp_family#gmlp) includes implementation of [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).

  | Model      | Params | FLOPs  | Input | Top1 Acc | Download |
  | ---------- | ------ | ------ | ----- | -------- | -------- |
  | GMLPTiny16 | 6M     | 1.35G  | 224   | 72.3     |          |
  | GMLPS16    | 20M    | 4.44G  | 224   | 79.6     | [gmlp_s16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/gmlp_s16_imagenet.h5) |
  | GMLPB16    | 73M    | 15.82G | 224   | 81.6     |          |
## GPViT
  - [Keras GPViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/gpvit) includes implementation of [PDF 2212.06795 GPVIT: A HIGH RESOLUTION NON-HIERARCHICAL VISION TRANSFORMER WITH GROUP PROPAGATION](https://arxiv.org/pdf/2212.06795.pdf).

  | Model    | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------- | ------ | ------ | ----- | -------- | -------- |
  | GPViT_L1 | 9.59M  | 6.15G  | 224   | 80.5     | [gpvit_l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l1_224_imagenet.h5) |
  | GPViT_L2 | 24.2M  | 15.74G | 224   | 83.4     | [gpvit_l2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l2_224_imagenet.h5) |
  | GPViT_L3 | 36.7M  | 23.54G | 224   | 84.1     | [gpvit_l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l3_224_imagenet.h5) |
  | GPViT_L4 | 75.5M  | 48.29G | 224   | 84.3     | [gpvit_l4_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l4_224_imagenet.h5) |
## HaloNet
  - [Keras HaloNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/halonet) is for [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).

  | Model          | Params | FLOPs   | Input | Top1 Acc | Download |
  | -------------- | ------ | ------- | ----- | -------- | -------- |
  | HaloNetH0      | 5.5M   | 2.40G   | 256   | 77.9     |          |
  | HaloNetH1      | 8.1M   | 3.04G   | 256   | 79.9     |          |
  | HaloNetH2      | 9.4M   | 3.37G   | 256   | 80.4     |          |
  | HaloNetH3      | 11.8M  | 6.30G   | 320   | 81.9     |          |
  | HaloNetH4      | 19.1M  | 12.17G  | 384   | 83.3     |          |
  | - 21k          | 19.1M  | 12.17G  | 384   | 85.5     |          |
  | HaloNetH5      | 30.7M  | 32.61G  | 448   | 84.0     |          |
  | HaloNetH6      | 43.4M  | 53.20G  | 512   | 84.4     |          |
  | HaloNetH7      | 67.4M  | 119.64G | 600   | 84.9     |          |
  | HaloNextECA26T | 10.7M  | 2.43G   | 256   | 79.50    | [halonext_eca26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonext_eca26t_256_imagenet.h5) |
  | HaloNet26T     | 12.5M  | 3.18G   | 256   | 79.13    | [halonet26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet26t_256_imagenet.h5) |
  | HaloNetSE33T   | 13.7M  | 3.55G   | 256   | 80.99    | [halonet_se33t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet_se33t_256_imagenet.h5) |
  | HaloRegNetZB   | 11.68M | 1.97G   | 224   | 81.042   | [haloregnetz_b_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/haloregnetz_b_224_imagenet.h5) |
  | HaloNet50T     | 22.7M  | 5.29G   | 256   | 81.70    | [halonet50t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet50t_256_imagenet.h5) |
  | HaloBotNet50T  | 22.6M  | 5.02G   | 256   | 82.0     | [halobotnet50t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halobotnet50t_256_imagenet.h5) |
## HorNet
  - [Keras HorNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/hornet) is for [PDF 2207.14284 HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/pdf/2207.14284.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | HorNetTiny    | 22.4M  | 4.01G  | 224   | 82.8     | [hornet_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_224_imagenet.h5) |
  | HorNetTinyGF  | 23.0M  | 3.94G  | 224   | 83.0     | [hornet_tiny_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_gf_224_imagenet.h5) |
  | HorNetSmall   | 49.5M  | 8.87G  | 224   | 83.8     | [hornet_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_224_imagenet.h5) |
  | HorNetSmallGF | 50.4M  | 8.77G  | 224   | 84.0     | [hornet_small_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_gf_224_imagenet.h5) |
  | HorNetBase    | 87.3M  | 15.65G | 224   | 84.2     | [hornet_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_224_imagenet.h5) |
  | HorNetBaseGF  | 88.4M  | 15.51G | 224   | 84.3     | [hornet_base_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_gf_224_imagenet.h5) |
  | HorNetLarge   | 194.5M | 34.91G | 224   | 86.8     | [hornet_large_224_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_224_imagenet22k.h5) |
  | HorNetLargeGF | 196.3M | 34.72G | 224   | 87.0     | [hornet_large_gf_224_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_224_imagenet22k.h5) |
  | HorNetLargeGF | 201.8M | 102.0G | 384   | 87.7     | [hornet_large_gf_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_384_imagenet22k.h5) |
## IFormer
  - [Keras IFormer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/iformer) is for [PDF 2205.12956 Inception Transformer](https://arxiv.org/pdf/2205.12956.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | IFormerSmall | 19.9M  | 4.88G  | 224   | 83.4     | [iformer_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_224_imagenet.h5) |
  |              | 20.9M  | 16.29G | 384   | 84.6     | [iformer_small_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_384_imagenet.h5) |
  | IFormerBase  | 47.9M  | 9.44G  | 224   | 84.6     | [iformer_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_224_imagenet.h5) |
  |              | 48.9M  | 30.86G | 384   | 85.7     | [iformer_base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_384_imagenet.h5) |
  | IFormerLarge | 86.6M  | 14.12G | 224   | 84.6     | [iformer_large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_224_imagenet.h5) |
  |              | 87.7M  | 45.74G | 384   | 85.8     | [iformer_large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_384_imagenet.h5) |
## InceptionNeXt
  - [Keras InceptionNeXt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/inceptionnext) is for [PDF 2303.16900 InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/pdf/2303.16900.pdf).

  | Model              | Params | FLOP s | Input | Top1 Acc | Download |
  | ------------------ | ------ | ------ | ----- | -------- | -------- |
  | InceptionNeXtTiny  | 28.05M | 4.21G  | 224   | 82.3     | [inceptionnext_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_tiny_imagenet.h5) |
  | InceptionNeXtSmall | 49.37M | 8.39G  | 224   | 83.5     | [inceptionnext_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_small_imagenet.h5) |
  | InceptionNeXtBase  | 86.67M | 14.88G | 224   | 84.0     | [inceptionnext_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_224_imagenet.h5) |
  |                    | 86.67M | 43.73G | 384   | 85.2     | [inceptionnext_base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_384_imagenet.h5) |
## LCNet
  - [Keras LCNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilenetv3_family#lcnet) includes implementation of [PDF 2109.15099 PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf).

  | Model    | Params | FLOPs   | Input | Top1 Acc | Download |
  | -------- | ------ | ------- | ----- | -------- | -------- |
  | LCNet050 | 1.88M  | 46.02M  | 224   | 63.10    | [lcnet_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_050_imagenet.h5) |
  | - ssld   | 1.88M  | 46.02M  | 224   | 66.10    | [lcnet_050_ssld.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_050_ssld.h5) |
  | LCNet075 | 2.36M  | 96.82M  | 224   | 68.82    | [lcnet_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_075_imagenet.h5) |
  | LCNet100 | 2.95M  | 158.28M | 224   | 72.10    | [lcnet_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_100_imagenet.h5) |
  | - ssld   | 2.95M  | 158.28M | 224   | 74.39    | [lcnet_100_ssld.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_100_ssld.h5) |
  | LCNet150 | 4.52M  | 338.05M | 224   | 73.71    | [lcnet_150_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_150_imagenet.h5) |
  | LCNet200 | 6.54M  | 585.35M | 224   | 75.18    | [lcnet_200_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_200_imagenet.h5) |
  | LCNet250 | 9.04M  | 900.16M | 224   | 76.60    | [lcnet_250_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_250_imagenet.h5) |
  | - ssld   | 9.04M  | 900.16M | 224   | 80.82    | [lcnet_250_ssld.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_250_ssld.h5) |
## LeViT
  - [Keras LeViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/levit) is for [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).

  | Model                   | Params | FLOPs | Input | Top1 Acc | Download |
  | ----------------------- | ------ | ----- | ----- | -------- | -------- |
  | LeViT128S, distillation | 7.8M   | 0.31G | 224   | 76.6     | [levit128s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128s_imagenet.h5) |
  | LeViT128, distillation  | 9.2M   | 0.41G | 224   | 78.6     | [levit128_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128_imagenet.h5) |
  | LeViT192, distillation  | 11M    | 0.66G | 224   | 80.0     | [levit192_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit192_imagenet.h5) |
  | LeViT256, distillation  | 19M    | 1.13G | 224   | 81.6     | [levit256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit256_imagenet.h5) |
  | LeViT384, distillation  | 39M    | 2.36G | 224   | 82.6     | [levit384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit384_imagenet.h5) |
## MaxViT
  - [Keras MaxViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/maxvit) is for [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).

  | Model                           | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------------- | ------ | ------ | ----- | -------- | -------- |
  | MaxViT_Tiny                     | 31M    | 5.6G   | 224   | 83.62    | [tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_224_imagenet.h5) |
  | MaxViT_Tiny                     | 31M    | 17.7G  | 384   | 85.24    | [tiny_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_384_imagenet.h5) |
  | MaxViT_Tiny                     | 31M    | 33.7G  | 512   | 85.72    | [tiny_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_512_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 11.7G  | 224   | 84.45    | [small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_224_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 36.1G  | 384   | 85.74    | [small_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_384_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 67.6G  | 512   | 86.19    | [small_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_512_imagenet.h5) |
  | MaxViT_Base                     | 119M   | 24.2G  | 224   | 84.95    | [base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet.h5) |
  | - imagenet21k                   | 135M   | 24.2G  | 224   |          | [base_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet21k.h5) |
  | MaxViT_Base                     | 119M   | 74.2G  | 384   | 86.34    | [base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet.h5) |
  | - imagenet21k-ft1k              | 119M   | 74.2G  | 384   | 88.24    | [base_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet21k-ft1k.h5) |
  | MaxViT_Base                     | 119M   | 138.5G | 512   | 86.66    | [base_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet.h5) |
  | - imagenet21k-ft1k              | 119M   | 138.5G | 512   | 88.38    | [base_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet21k-ft1k.h5) |
  | MaxViT_Large                    | 212M   | 43.9G  | 224   | 85.17    | [large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet.h5) |
  | - imagenet21k                   | 233M   | 43.9G  | 224   |          | [large_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet21k.h5) |
  | MaxViT_Large                    | 212M   | 133.1G | 384   | 86.40    | [large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet.h5) |
  | - imagenet21k-ft1k              | 212M   | 133.1G | 384   | 88.32    | [large_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet21k-ft1k.h5) |
  | MaxViT_Large                    | 212M   | 245.4G | 512   | 86.70    | [large_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet.h5) |
  | - imagenet21k-ft1k              | 212M   | 245.4G | 512   | 88.46    | [large_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet21k-ft1k.h5) |
  | MaxViT_XLarge, imagenet21k      | 507M   | 97.7G  | 224   |          | [xlarge_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_224_imagenet21k.h5) |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 293.7G | 384   | 88.51    | [xlarge_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_384_imagenet21k-ft1k.h5) |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 535.2G | 512   | 88.70    | [xlarge_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_512_imagenet21k-ft1k.h5) |
## MLP mixer
  - [Keras MLP mixer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mlp_family#mlp-mixer) includes implementation of [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).

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
## MobileNetV3
  - [Keras MobileNetV3](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilenetv3_family#mobilenetv3) includes implementation of [PDF 1905.02244 Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

  | Model               | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------------- | ------ | ------- | ----- | -------- | -------- |
  | MobileNetV3Small050 | 1.29M  | 24.92M  | 224   | 57.89    | [small_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_050_imagenet.h5) |
  | MobileNetV3Small075 | 2.04M  | 44.35M  | 224   | 65.24    | [small_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_075_imagenet.h5) |
  | MobileNetV3Small100 | 2.54M  | 57.62M  | 224   | 67.66    | [small_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_100_imagenet.h5) |
  | MobileNetV3Large075 | 3.99M  | 156.30M | 224   | 73.44    | [large_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_075_imagenet.h5) |
  | MobileNetV3Large100 | 5.48M  | 218.73M | 224   | 75.77    | [large_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_imagenet.h5) |
  | - miil              | 5.48M  | 218.73M | 224   | 77.92    | [large_100_miil.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_mill.h5) |
## MobileViT
  - [Keras MobileViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilevit) is for [PDF 2110.02178 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER](https://arxiv.org/pdf/2110.02178.pdf).

  | Model         | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------- | ------ | ----- | ----- | -------- | -------- |
  | MobileViT_XXS | 1.3M   | 0.42G | 256   | 69.0     | [mobilevit_xxs_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xxs_imagenet.h5) |
  | MobileViT_XS  | 2.3M   | 1.05G | 256   | 74.7     | [mobilevit_xs_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xs_imagenet.h5) |
  | MobileViT_S   | 5.6M   | 2.03G | 256   | 78.3     | [mobilevit_s_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_s_imagenet.h5) |
## MobileViT_V2
  - [Keras MobileViT_V2](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilevit) is for [PDF 2206.02680 Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/pdf/2206.02680.pdf).

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | MobileViT_V2_050   | 1.37M  | 0.47G | 256   | 70.18    | [v2_050_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_050_256_imagenet.h5) |
  | MobileViT_V2_075   | 2.87M  | 1.04G | 256   | 75.56    | [v2_075_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_075_256_imagenet.h5) |
  | MobileViT_V2_100   | 4.90M  | 1.83G | 256   | 78.09    | [v2_100_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_100_256_imagenet.h5) |
  | MobileViT_V2_125   | 7.48M  | 2.84G | 256   | 79.65    | [v2_125_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_125_256_imagenet.h5) |
  | MobileViT_V2_150   | 10.6M  | 4.07G | 256   | 80.38    | [v2_150_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet.h5) |
  | - imagenet22k      | 10.6M  | 4.07G | 256   | 81.46    | [v2_150_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 10.6M  | 9.15G | 384   | 82.60    | [v2_150_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_384_imagenet22k.h5) |
  | MobileViT_V2_175   | 14.3M  | 5.52G | 256   | 80.84    | [v2_175_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet.h5) |
  | - imagenet22k      | 14.3M  | 5.52G | 256   | 81.94    | [v2_175_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 14.3M  | 12.4G | 384   | 82.93    | [v2_175_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_384_imagenet22k.h5) |
  | MobileViT_V2_200   | 18.4M  | 7.12G | 256   | 81.17    | [v2_200_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet.h5) |
  | - imagenet22k      | 18.4M  | 7.12G | 256   | 82.36    | [v2_200_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 18.4M  | 16.2G | 384   | 83.41    | [v2_200_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_384_imagenet22k.h5) |
## MogaNet
  - [Keras MogaNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/moganet) is for [PDF 2211.03295 Efficient Multi-order Gated Aggregation Network](https://arxiv.org/pdf/2211.03295.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | MogaNetXtiny | 2.96M  | 806M   | 224   | 76.5     | [moganet_xtiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_xtiny_imagenet.h5) |
  | MogaNetTiny  | 5.20M  | 1.11G  | 224   | 79.0     | [moganet_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_224_imagenet.h5) |
  |              | 5.20M  | 1.45G  | 256   | 79.6     | [moganet_tiny_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_256_imagenet.h5) |
  | MogaNetSmall | 25.3M  | 4.98G  | 224   | 83.4     | [moganet_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_small_imagenet.h5) |
  | MogaNetBase  | 43.7M  | 9.96G  | 224   | 84.2     | [moganet_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_base_imagenet.h5) |
  | MogaNetLarge | 82.5M  | 15.96G | 224   | 84.6     | [moganet_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_large_imagenet.h5) |
## NAT
  - [Keras NAT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/nat) is for [PDF 2204.07143 Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf).

  | Model     | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------- | ------ | ------ | ----- | -------- | -------- |
  | NAT_Mini  | 20.0M  | 2.73G  | 224   | 81.8     | [nat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_mini_imagenet.h5) |
  | NAT_Tiny  | 27.9M  | 4.34G  | 224   | 83.2     | [nat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_tiny_imagenet.h5) |
  | NAT_Small | 50.7M  | 7.84G  | 224   | 83.7     | [nat_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_small_imagenet.h5) |
  | NAT_Base  | 89.8M  | 13.76G | 224   | 84.3     | [nat_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_base_imagenet.h5) |
## NFNets
  - [Keras NFNets](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/nfnets) is for [PDF 2102.06171 High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171.pdf).

  | Model       | Params | FLOPs   | Input | Top1 Acc | Download |
  | ----------- | ------ | ------- | ----- | -------- | -------- |
  | NFNetL0     | 35.07M | 7.13G   | 288   | 82.75    | [nfnetl0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetl0_imagenet.h5) |
  | NFNetF0     | 71.5M  | 12.58G  | 256   | 83.6     | [nfnetf0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf0_imagenet.h5) |
  | NFNetF1     | 132.6M | 35.95G  | 320   | 84.7     | [nfnetf1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf1_imagenet.h5) |
  | NFNetF2     | 193.8M | 63.24G  | 352   | 85.1     | [nfnetf2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf2_imagenet.h5) |
  | NFNetF3     | 254.9M | 115.75G | 416   | 85.7     | [nfnetf3_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf3_imagenet.h5) |
  | NFNetF4     | 316.1M | 216.78G | 512   | 85.9     | [nfnetf4_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf4_imagenet.h5) |
  | NFNetF5     | 377.2M | 291.73G | 544   | 86.0     | [nfnetf5_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf5_imagenet.h5) |
  | NFNetF6 SAM | 438.4M | 379.75G | 576   | 86.5     | [nfnetf6_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf6_imagenet.h5) |
  | NFNetF7     | 499.5M | 481.80G | 608   |          |          |
  | ECA_NFNetL0 | 24.14M | 7.12G   | 288   | 82.58    | [eca_nfnetl0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl0_imagenet.h5) |
  | ECA_NFNetL1 | 41.41M | 14.93G  | 320   | 84.01    | [eca_nfnetl1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl1_imagenet.h5) |
  | ECA_NFNetL2 | 56.72M | 30.12G  | 384   | 84.70    | [eca_nfnetl2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl2_imagenet.h5) |
  | ECA_NFNetL3 | 72.04M | 52.73G  | 448   |          |          |
## PVT_V2
  - [Keras PVT_V2](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/pvt) is for [PDF 2106.13797 PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/pdf/2106.13797.pdf).

  | Model           | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------------- | ------ | ------ | ----- | -------- | -------- |
  | PVT_V2B0        | 3.7M   | 580.3M | 224   | 70.5     | [pvt_v2_b0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b0_imagenet.h5) |
  | PVT_V2B1        | 14.0M  | 2.14G  | 224   | 78.7     | [pvt_v2_b1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b1_imagenet.h5) |
  | PVT_V2B2        | 25.4M  | 4.07G  | 224   | 82.0     | [pvt_v2_b2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_imagenet.h5) |
  | PVT_V2B2_linear | 22.6M  | 3.94G  | 224   | 82.1     | [pvt_v2_b2_linear.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_linear_imagenet.h5) |
  | PVT_V2B3        | 45.2M  | 6.96G  | 224   | 83.1     | [pvt_v2_b3_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b3_imagenet.h5) |
  | PVT_V2B4        | 62.6M  | 10.19G | 224   | 83.6     | [pvt_v2_b4_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b4_imagenet.h5) |
  | PVT_V2B5        | 82.0M  | 11.81G | 224   | 83.8     | [pvt_v2_b5_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b5_imagenet.h5) |
## RegNetY
  - [Keras RegNetY](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnet_family#regnety) is for [PDF 2003.13678 Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf).

  | Model      | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ---------- | ------- | ------ | ----- | -------- | -------- |
  | RegNetY040 | 20.65M  | 3.98G  | 224   | 82.3     | [regnety_040_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_040_imagenet.h5) |
  | RegNetY064 | 30.58M  | 6.36G  | 224   | 83.0     | [regnety_064_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_064_imagenet.h5) |
  | RegNetY080 | 39.18M  | 7.97G  | 224   | 83.17    | [regnety_080_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_080_imagenet.h5) |
  | RegNetY160 | 83.59M  | 15.92G | 224   | 82.0     | [regnety_160_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_160_imagenet.h5) |
  | RegNetY320 | 145.05M | 32.29G | 224   | 82.5     | [regnety_320_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_320_imagenet.h5) |
## RegNetZ
  - [Keras RegNetZ](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnet_family#regnetz) includes implementation of [Github timm/models/byobnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py).
  - Related paper [PDF 2004.02967 Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf)

  | Model          | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------- | ------ | ----- | ----- | -------- | -------- |
  | RegNetZB16     | 9.72M  | 1.44G | 224   | 79.868   | [regnetz_b16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_b16_imagenet.h5) |
  | RegNetZC16     | 13.46M | 2.50G | 256   | 82.164   | [regnetz_c16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_c16_imagenet.h5) |
  | RegNetZC16_EVO | 13.49M | 2.55G | 256   | 81.9     | [regnetz_c16_evo_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_c16_evo_imagenet.h5) |
  | RegNetZD32     | 27.58M | 5.96G | 256   | 83.422   | [regnetz_d32_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d32_imagenet.h5) |
  | RegNetZD8      | 23.37M | 3.95G | 256   | 83.5     | [regnetz_d8_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d8_imagenet.h5)   |
  | RegNetZD8_EVO  | 23.46M | 4.61G | 256   | 83.42    | [regnetz_d8_evo_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d8_evo_imagenet.h5)   |
  | RegNetZE8      | 57.70M | 9.88G | 256   | 84.5     | [regnetz_e8_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_e8_imagenet.h5)   |
## ResMLP
  - [Keras ResMLP](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mlp_family#resmlp) includes implementation of [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf).

  | Model         | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------- | ------ | ------- | ----- | -------- | -------- |
  | ResMLP12      | 15M    | 3.02G   | 224   | 77.8     | [resmlp12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp12_imagenet.h5) |
  | ResMLP24      | 30M    | 5.98G   | 224   | 80.8     | [resmlp24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp24_imagenet.h5) |
  | ResMLP36      | 116M   | 8.94G   | 224   | 81.1     | [resmlp36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp36_imagenet.h5) |
  | ResMLP_B24    | 129M   | 100.39G | 224   | 83.6     | [resmlp_b24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet.h5) |
  | - imagenet22k | 129M   | 100.39G | 224   | 84.4     | [resmlp_b24_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet22k.h5) |
## ResNeSt
  - [Keras ResNeSt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnest) is for [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf).

  | Model          | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------------- | ------ | ------ | ----- | -------- | -------- |
  | resnest50      | 28M    | 5.38G  | 224   | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50_imagenet.h5) |
  | resnest101     | 49M    | 13.33G | 256   | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101_imagenet.h5) |
  | resnest200     | 71M    | 35.55G | 320   | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200_imagenet.h5) |
  | resnest269     | 111M   | 77.42G | 416   | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269_imagenet.h5) |
## ResNetD
  - [Keras ResNetD](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnet_family#resnetd) includes implementation of [PDF 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

  | Model      | Params | FLOPs  | Input | Top1 Acc | Download |
  | ---------- | ------ | ------ | ----- | -------- | -------- |
  | ResNet50D  | 25.58M | 4.33G  | 224   | 80.530   | [resnet50d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet50d_imagenet.h5) |
  | ResNet101D | 44.57M | 8.04G  | 224   | 83.022   | [resnet101d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet101d_imagenet.h5) |
  | ResNet152D | 60.21M | 11.75G | 224   | 83.680   | [resnet152d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet152d_imagenet.h5) |
  | ResNet200D | 64.69M | 15.25G | 224   | 83.962   | [resnet200d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet200d_imagenet.h5) |
## ResNetQ
  - [Keras ResNetQ](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnet_family#resnetq) includes implementation of [Github timm/models/resnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py)

  | Model     | Params | FLOPs | Input | Top1 Acc | Download |
  | --------- | ------ | ----- | ----- | -------- | -------- |
  | ResNet51Q | 35.7M  | 4.87G | 224   | 82.36    | [resnet51q.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet51q_imagenet.h5) |
  | ResNet61Q | 36.8M  | 5.96G | 224   |          |          |
## ResNeXt
  - [Keras ResNeXt](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/resnet_family#resnext) includes implementation of [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
  - `SWSL` means `Semi-Weakly Supervised ResNe*t` from [Github facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models). **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.

  | Model                     | Params | FLOPs  | Input | Top1 Acc | Download            |
  | ------------------------- | ------ | ------ | ----- | -------- | ------------------- |
  | ResNeXt50 (32x4d)         | 25M    | 4.23G  | 224   | 79.768   | [resnext50_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_imagenet.h5)  |
  | - SWSL                    | 25M    | 4.23G  | 224   | 82.182   | [resnext50_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_swsl.h5)  |
  | ResNeXt50D (32x4d + deep) | 25M    | 4.47G  | 224   | 79.676   | [resnext50d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50d_imagenet.h5)  |
  | ResNeXt101 (32x4d)        | 42M    | 7.97G  | 224   | 80.334   | [resnext101_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)  |
  | - SWSL                    | 42M    | 7.97G  | 224   | 83.230   | [resnext101_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_swsl.h5)  |
  | ResNeXt101W (32x8d)       | 89M    | 16.41G | 224   | 79.308   | [resnext101_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)  |
  | - SWSL                    | 89M    | 16.41G | 224   | 84.284   | [resnext101w_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101w_swsl.h5)  |
  | ResNeXt101W_64 (64x4d)    | 83.46M | 15.46G | 224   | 82.46    | [resnext101w_64_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101w_64_imagenet.h5)  |
## SwinTransformerV2
  - [Keras SwinTransformerV2](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/swin_transformer_v2) includes implementation of [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).

  | Model                                | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------------------ | ------ | ------ | ----- | -------- | -------- |
  | SwinTransformerV2Tiny_ns             | 28.3M  | 4.69G  | 224   | 81.8     | [tiny_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_ns_224_imagenet.h5) |
  | SwinTransformerV2Small_ns            | 49.7M  | 9.12G  | 224   | 83.5     | [small_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_ns_224_imagenet.h5) |
  |                                      |        |        |       |          |          |
  | SwinTransformerV2Tiny_window8        | 28.3M  | 5.99G  | 256   | 81.8     | [tiny_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window8_256_imagenet.h5) |
  | SwinTransformerV2Tiny_window16       | 28.3M  | 6.75G  | 256   | 82.8     | [tiny_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window16_256_imagenet.h5) |
  | SwinTransformerV2Small_window8       | 49.7M  | 11.63G | 256   | 83.7     | [small_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window8_256_imagenet.h5) |
  | SwinTransformerV2Small_window16      | 49.7M  | 12.93G | 256   | 84.1     | [small_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window16_256_imagenet.h5) |
  | SwinTransformerV2Base_window8        | 87.9M  | 20.44G | 256   | 84.2     | [base_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window8_256_imagenet.h5) |
  | SwinTransformerV2Base_window16       | 87.9M  | 22.17G | 256   | 84.6     | [base_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet.h5) |
  | SwinTransformerV2Base_window16, 22k  | 87.9M  | 22.17G | 256   | 86.2     | [base_window16_256_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet22k.h5) |
  | SwinTransformerV2Base_window24, 22k  | 87.9M  | 55.89G | 384   | 87.1     | [base_window24_384_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window24_384_imagenet22k.h5) |
  | SwinTransformerV2Large_window16, 22k | 196.7M | 48.03G | 256   | 86.9     | [large_window16_256_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window16_256_imagenet22k.h5) |
  | SwinTransformerV2Large_window24, 22k | 196.7M | 117.1G | 384   | 87.6     | [large_window24_384_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window24_384_imagenet22k.h5) |
## TinyNet
  - [Keras TinyNet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mobilenetv3_family#tinynet) includes implementation of [PDF 2010.14819 Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets](https://arxiv.org/pdf/2010.14819.pdf).

  | Model    | Params | FLOPs   | Input | Top1 Acc | Download |
  | -------- | ------ | ------- | ----- | -------- | -------- |
  | TinyNetE | 2.04M  | 25.22M  | 106   | 59.86    | [tinynet_e_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_e_imagenet.h5) |
  | TinyNetD | 2.34M  | 53.35M  | 152   | 66.96    | [tinynet_d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_d_imagenet.h5) |
  | TinyNetC | 2.46M  | 103.22M | 184   | 71.23    | [tinynet_c_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_c_imagenet.h5) |
  | TinyNetB | 3.73M  | 206.28M | 188   | 74.98    | [tinynet_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_b_imagenet.h5) |
  | TinyNetA | 6.19M  | 343.74M | 192   | 77.65    | [tinynet_a_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_a_imagenet.h5) |
## TinyViT
  - [Keras TinyViT](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/tinyvit) includes implementation of [PDF 2207.10666 TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf).

  | Model                | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------------- | ------ | ----- | ----- | -------- | -------- |
  | TinyViT_5M, distill  | 5.4M   | 1.3G  | 224   | 79.1     | [tiny_vit_5m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet.h5) |
  | - imagenet21k-ft1k   | 5.4M   | 1.3G  | 224   | 80.7     | [tiny_vit_5m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet21k-ft1k.h5) |
  | TinyViT_11M, distill | 11M    | 2.0G  | 224   | 81.5     | [tiny_vit_11m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet.h5) |
  | - imagenet21k-ft1k   | 11M    | 2.0G  | 224   | 83.2     | [tiny_vit_11m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet21k-ft1k.h5) |
  | TinyViT_21M, distill | 21M    | 4.3G  | 224   | 83.1     | [tiny_vit_21m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet.h5) |
  | - imagenet21k-ft1k   | 21M    | 4.3G  | 224   | 84.8     | [tiny_vit_21m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet21k-ft1k.h5) |
  |                      | 21M    | 13.8G | 384   | 86.2     | [tiny_vit_21m_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_384_imagenet21k-ft1k.h5) |
  |                      | 21M    | 27.0G | 512   | 86.5     | [tiny_vit_21m_512_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_512_imagenet21k-ft1k.h5) |
## UniFormer
  - [Keras UniFormer](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/uniformer) includes implementation of [PDF 2201.09450 UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/pdf/2201.09450.pdf).

  | Model                 | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------------------- | ------ | ------ | ----- | -------- | -------- |
  | UniformerSmall32 + TL | 22M    | 3.66G  | 224   | 83.4     | [small_32_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_32_224_token_label.h5) |
  | UniformerSmall64      | 22M    | 3.66G  | 224   | 82.9     | [small_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_imagenet.h5) |
  | - Token Labeling      | 22M    | 3.66G  | 224   | 83.4     | [small_64_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_token_label.h5) |
  | UniformerSmallPlus32  | 24M    | 4.24G  | 224   | 83.4     | [small_plus_32_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 4.24G  | 224   | 83.9     | [small_plus_32_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_token_label.h5) |
  | UniformerSmallPlus64  | 24M    | 4.23G  | 224   | 83.4     | [small_plus_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 4.23G  | 224   | 83.6     | [small_plus_64_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_token_label.h5) |
  | UniformerBase32 + TL  | 50M    | 8.32G  | 224   | 85.1     | [base_32_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_32_224_token_label.h5) |
  | UniformerBase64       | 50M    | 8.31G  | 224   | 83.8     | [base_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_imagenet.h5) |
  | - Token Labeling      | 50M    | 8.31G  | 224   | 84.8     | [base_64_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_token_label.h5) |
  | UniformerLarge64 + TL | 100M   | 19.79G | 224   | 85.6     | [large_64_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_224_token_label.h5) |
  | UniformerLarge64 + TL | 100M   | 63.11G | 384   | 86.3     | [large_64_384_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_384_token_label.h5) |
## VOLO
  - [Keras VOLO](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/volo) is for [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).

  | Model   | Params | FLOPs   | Input | Top1 Acc | Download            |
  | ------- | ------ | ------- | ----- | -------- | ------------------- |
  | VOLO_d1 | 27M    | 4.82G   | 224   | 84.2     | [volo_d1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_224_imagenet.h5) |
  | - 384   | 27M    | 14.22G  | 384   | 85.2     | [volo_d1_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_384_imagenet.h5) |
  | VOLO_d2 | 59M    | 9.78G   | 224   | 85.2     | [volo_d2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_224_imagenet.h5) |
  | - 384   | 59M    | 28.84G  | 384   | 86.0     | [volo_d2_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_384_imagenet.h5) |
  | VOLO_d3 | 86M    | 13.80G  | 224   | 85.4     | [volo_d3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_224_imagenet.h5) |
  | - 448   | 86M    | 55.50G  | 448   | 86.3     | [volo_d3_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_448_imagenet.h5) |
  | VOLO_d4 | 193M   | 29.39G  | 224   | 85.7     | [volo_d4_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_224_imagenet.h5) |
  | - 448   | 193M   | 117.81G | 448   | 86.8     | [volo_d4_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_448_imagenet.h5) |
  | VOLO_d5 | 296M   | 53.34G  | 224   | 86.1     | [volo_d5_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_224_imagenet.h5) |
  | - 448   | 296M   | 213.72G | 448   | 87.0     | [volo_d5_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_448_imagenet.h5) |
  | - 512   | 296M   | 279.36G | 512   | 87.1     | [volo_d5_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_512_imagenet.h5) |
## WaveMLP
  - [Keras WaveMLP](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/mlp_family#wavemlp) includes implementation of [PDF 2111.12294 An Image Patch is a Wave: Quantum Inspired Vision MLP](https://arxiv.org/pdf/2111.12294.pdf).

  | Model     | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------- | ------ | ------ | ----- | -------- | -------- |
  | WaveMLP_T | 17M    | 2.47G  | 224   | 80.9     | [wavemlp_t_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_t_imagenet.h5) |
  | WaveMLP_S | 30M    | 4.55G  | 224   | 82.9     | [wavemlp_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_s_imagenet.h5) |
  | WaveMLP_M | 44M    | 7.92G  | 224   | 83.3     | [wavemlp_m_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_m_imagenet.h5) |
  | WaveMLP_B | 63M    | 10.26G | 224   | 83.6     |          |
***

# Detection Models
## EfficientDet
  - [Keras EfficientDet](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/efficientdet) includes implementation of [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).
  - `Det-AdvProp + AutoAugment` [Paper 2103.13886 Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/pdf/2103.13886.pdf).

  | Model              | Params | FLOPs   | Input | COCO val AP | test AP | Download |
  | ------------------ | ------ | ------- | ----- | ----------- | ------- | -------- |
  | EfficientDetD0     | 3.9M   | 2.55G   | 512   | 34.3        | 34.6    | [efficientdet_d0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d0_512_coco.h5)         |
  | - Det-AdvProp      | 3.9M   | 2.55G   | 512   | 35.1        | 35.3    |          |
  | EfficientDetD1     | 6.6M   | 6.13G   | 640   | 40.2        | 40.5    | [efficientdet_d1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d1_640_coco.h5)         |
  | - Det-AdvProp      | 6.6M   | 6.13G   | 640   | 40.8        | 40.9    |          |
  | EfficientDetD2     | 8.1M   | 11.03G  | 768   | 43.5        | 43.9    | [efficientdet_d2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d2_768_coco.h5)         |
  | - Det-AdvProp      | 8.1M   | 11.03G  | 768   | 44.3        | 44.3    |          |
  | EfficientDetD3     | 12.0M  | 24.95G  | 896   | 46.8        | 47.2    | [efficientdet_d3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d3_896_coco.h5)         |
  | - Det-AdvProp      | 12.0M  | 24.95G  | 896   | 47.7        | 48.0    |          |
  | EfficientDetD4     | 20.7M  | 55.29G  | 1024  | 49.3        | 49.7    | [efficientdet_d4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d4_1024_coco.h5)        |
  | - Det-AdvProp      | 20.7M  | 55.29G  | 1024  | 50.4        | 50.4    |          |
  | EfficientDetD5     | 33.7M  | 135.62G | 1280  | 51.2        | 51.5    | [efficientdet_d5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d5_1280_coco.h5)        |
  | - Det-AdvProp      | 33.7M  | 135.62G | 1280  | 52.2        | 52.5    |          |
  | EfficientDetD6     | 51.9M  | 225.93G | 1280  | 52.1        | 52.6    | [efficientdet_d6.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d6_1280_coco.h5)        |
  | EfficientDetD7     | 51.9M  | 325.34G | 1536  | 53.4        | 53.7    | [efficientdet_d7.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7_1536_coco.h5)        |
  | EfficientDetD7X    | 77.0M  | 410.87G | 1536  | 54.4        | 55.1    | [efficientdet_d7x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7x_1536_coco.h5)      |
  | EfficientDetLite0  | 3.2M   | 0.98G   | 320   | 27.5        | 26.41   | [efficientdet_lite0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite0_320_coco.h5)   |
  | EfficientDetLite1  | 4.2M   | 1.97G   | 384   | 32.6        | 31.50   | [efficientdet_lite1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite1_384_coco.h5)   |
  | EfficientDetLite2  | 5.3M   | 3.38G   | 448   | 36.2        | 35.06   | [efficientdet_lite2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite2_448_coco.h5)   |
  | EfficientDetLite3  | 8.4M   | 7.50G   | 512   | 39.9        | 38.77   | [efficientdet_lite3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3_512_coco.h5)   |
  | EfficientDetLite3X | 9.3M   | 14.01G  | 640   | 44.0        | 42.64   | [efficientdet_lite3x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3x_640_coco.h5) |
  | EfficientDetLite4  | 15.1M  | 20.20G  | 640   | 44.4        | 43.18   | [efficientdet_lite4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite4_640_coco.h5)   |
## YOLOR
  - [Keras YOLOR](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/yolor) includes implementation of [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).

  | Model      | Params | FLOPs   | Input | COCO val AP | test AP | Download |
  | ---------- | ------ | ------- | ----- | ----------- | ------- | -------- |
  | YOLOR_CSP  | 52.9M  | 60.25G  | 640   | 50.0        | 52.8    | [yolor_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_coco.h5)     |
  | YOLOR_CSPX | 99.8M  | 111.11G | 640   | 51.5        | 54.8    | [yolor_csp_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_x_coco.h5) |
  | YOLOR_P6   | 37.3M  | 162.87G | 1280  | 52.5        | 55.7    | [yolor_p6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_p6_coco.h5)       |
  | YOLOR_W6   | 79.9M  | 226.67G | 1280  | 53.6 ?      | 56.9    | [yolor_w6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_w6_coco.h5)       |
  | YOLOR_E6   | 115.9M | 341.62G | 1280  | 50.3 ?      | 57.6    | [yolor_e6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_e6_coco.h5)       |
  | YOLOR_D6   | 151.8M | 467.88G | 1280  | 50.8 ?      | 58.2    | [yolor_d6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_d6_coco.h5)       |
## YOLOV7
  - [Keras YOLOV7](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/yolov7) includes implementation of [Paper 2207.02696 YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf).

  | Model       | Params | FLOPs  | Input | COCO val AP | test AP | Download |
  | ----------- | ------ | ------ | ----- | ----------- | ------- | -------- |
  | YOLOV7_Tiny | 6.23M  | 2.90G  | 416   | 33.3        |         | [yolov7_tiny_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_tiny_coco.h5) |
  | YOLOV7_CSP  | 37.67M | 53.0G  | 640   | 51.4        |         | [yolov7_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_csp_coco.h5) |
  | YOLOV7_X    | 71.41M | 95.0G  | 640   | 53.1        |         | [yolov7_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_x_coco.h5) |
  | YOLOV7_W6   | 70.49M | 180.1G | 1280  | 54.9        |         | [yolov7_w6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_w6_coco.h5) |
  | YOLOV7_E6   | 97.33M | 257.6G | 1280  | 56.0        |         | [yolov7_e6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6_coco.h5) |
  | YOLOV7_D6   | 133.9M | 351.4G | 1280  | 56.6        |         | [yolov7_d6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_d6_coco.h5) |
  | YOLOV7_E6E  | 151.9M | 421.7G | 1280  | 56.8        |         | [yolov7_e6e_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6e_coco.h5) |
## YOLOX
  - [Keras YOLOX](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/yolox) includes implementation of [Paper 2107.08430 YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf).

  | Model     | Params | FLOPs   | Input | COCO val AP | test AP | Download |
  | --------- | ------ | ------- | ----- | ----------- | ------- | -------- |
  | YOLOXNano | 0.91M  | 0.53G   | 416   | 25.8        |         | [yolox_nano_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_nano_coco.h5) |
  | YOLOXTiny | 5.06M  | 3.22G   | 416   | 32.8        |         | [yolox_tiny_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_tiny_coco.h5) |
  | YOLOXS    | 9.0M   | 13.39G  | 640   | 40.5        | 40.5    | [yolox_s_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_s_coco.h5)       |
  | YOLOXM    | 25.3M  | 36.84G  | 640   | 46.9        | 47.2    | [yolox_m_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_m_coco.h5)       |
  | YOLOXL    | 54.2M  | 77.76G  | 640   | 49.7        | 50.1    | [yolox_l_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_l_coco.h5)       |
  | YOLOXX    | 99.1M  | 140.87G | 640   | 51.5        | 51.5    | [yolox_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_x_coco.h5)       |
***

# Licenses
  - This part is copied and modified according to [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
  - **Code**. The code here is licensed MIT. It is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue. So far all of the pretrained weights available here are pretrained on ImageNet and COCO with a select few that have some additional pretraining.
  - **ImageNet Pretrained Weights**. ImageNet was released for non-commercial research purposes only (https://image-net.org/download). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.
  - **COCO Pretrained Weights**. Should follow [cocodataset termsofuse](https://cocodataset.org/#termsofuse). The annotations in COCO dataset belong to the COCO Consortium and are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode). The COCO Consortium does not own the copyright of the images. Use of the images must abide by the [Flickr Terms of Use](https://www.flickr.com/creativecommons/). The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset.
  - **Pretrained on more than ImageNet and COCO**. Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.
***

# Citing
  - **BibTeX**
    ```bibtex
    @misc{leondgarse,
      author = {Leondgarse},
      title = {Keras CV Attention Models},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      doi = {10.5281/zenodo.6506947},
      howpublished = {\url{https://github.com/leondgarse/keras_cv_attention_models}}
    }
    ```
  - **Latest DOI**: [![DOI](https://zenodo.org/badge/391777965.svg)](https://zenodo.org/badge/latestdoi/391777965)
***
