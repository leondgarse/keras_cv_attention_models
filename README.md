# ___Keras_cv_attention_models___
***
- **coco_train_script.py is under testing. Still struggling for this...**
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___>>>> Roadmap and todo list <<<<___](https://github.com/leondgarse/keras_cv_attention_models/wiki/Roadmap)
- [General Usage](#general-usage)
  - [Basic](#basic)
  - [T4 Inference](#t4-inference)
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
  - [DiNAT](#dinat)
  - [DINOv2](#dinov2)
  - [EdgeNeXt](#edgenext)
  - [EfficientFormer](#efficientformer)
  - [EfficientFormerV2](#efficientformerv2)
  - [EfficientNet](#efficientnet)
  - [EfficientNetV2](#efficientnetv2)
  - [EfficientViT_B](#efficientvit_b)
  - [EfficientViT_M](#efficientvit_m)
  - [EVA](#eva)
  - [EVA02](#eva02)
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
  - [VanillaNet](#vanillanet)
  - [VOLO](#volo)
  - [WaveMLP](#wavemlp)
- [Detection Models](#detection-models)
  - [EfficientDet](#efficientdet)
  - [YOLO_NAS](#yolo_nas)
  - [YOLOR](#yolor)
  - [YOLOV7](#yolov7)
  - [YOLOV8](#yolov8)
  - [YOLOX](#yolox)
- [Language Models](#language-models)
  - [GPT2](#gpt2)
- [Licenses](#licenses)
- [Citing](#citing)

<!-- /TOC -->
***

# General Usage
## Basic
  - **Currently recommended TF version is `tensorflow==2.11.1`. Expecially for training or TFLite conversion**.
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
  - Install as pip package. `kecam` is a short alias name of this package. **Note**: the pip package `kecam` doesn't set any backend requirement, make sure either Tensorflow or PyTorch installed before hand. For PyTorch backend usage, refer [Keras PyTorch Backend](keras_cv_attention_models/pytorch_backend).
    ```sh
    pip install -U kecam
    # Or
    pip install -U keras-cv-attention-models
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
    from keras_cv_attention_models.test_images import cat
    img = cat()
    imm = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.99664897),
    #  ('n02123045', 'tabby', 0.0007249644),
    #  ('n02123159', 'tiger_cat', 0.00020345),
    #  ('n02127052', 'lynx', 5.4973923e-05),
    #  ('n02123597', 'Siamese_cat', 2.675306e-05)]
    ```
    Or just use model preset `preprocess_input` and `decode_predictions`
    ```py
    from keras_cv_attention_models import coatnet
    mm = coatnet.CoAtNet0()

    from keras_cv_attention_models.test_images import cat
    preds = mm(mm.preprocess_input(cat()))
    print(mm.decode_predictions(preds))
    # [[('n02124075', 'Egyptian_cat', 0.9999875), ('n02123045', 'tabby', 5.194884e-06), ...]]
    ```
    The preset `preprocess_input` and `decode_predictions` also compatible with PyTorch backend.
    ```py
    os.environ['KECAM_BACKEND'] = 'torch'

    from keras_cv_attention_models import caformer
    mm = caformer.CAFormerS18()
    # >>>> Using PyTorch backend
    # >>>> Aligned input_shape: [3, 224, 224]
    # >>>> Load pretrained from: ~/.keras/models/caformer_s18_224_imagenet.h5

    from keras_cv_attention_models.test_images import cat
    preds = mm(mm.preprocess_input(cat()))
    print(preds.shape)
    # torch.Size([1, 1000])
    print(mm.decode_predictions(preds))
    # [[('n02124075', 'Egyptian_cat', 0.8817097), ('n02123045', 'tabby', 0.009335292), ...]]
    ```
  - **`num_classes=0`** set for excluding model top `GlobalAveragePooling2D + Dense` layers.
    ```py
    from keras_cv_attention_models import resnest
    mm = resnest.ResNest50(num_classes=0)
    print(mm.output_shape)
    # (None, 7, 7, 2048)
    ```
  - **`num_classes={custom output classes}`** others than `1000` or `0` will just skip loading the header Dense layer weights. As `model.load_weights(weight_file, by_name=True, skip_mismatch=True)` is used for loading weights.
    ```py
    from keras_cv_attention_models import swin_transformer_v2

    mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(num_classes=64)
    # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_window8_256_imagenet.h5
    # WARNING:tensorflow:Skipping loading weights for layer #601 (named predictions) due to mismatch in shape for weight predictions/kernel:0. Weight expects shape (768, 64). Received saved weight with shape (768, 1000)
    # WARNING:tensorflow:Skipping loading weights for layer #601 (named predictions) due to mismatch in shape for weight predictions/bias:0. Weight expects shape (64,). Received saved weight with shape (1000,)
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
  - **Calculate flops** method from [TF 2.0 Feature: Flops calculation #32809](https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-849439287). For PyTorch backend, needs `thop` `pip install thop`.
    ```py
    from keras_cv_attention_models import coatnet, resnest, model_surgery

    model_surgery.get_flops(coatnet.CoAtNet0())
    # >>>> FLOPs: 4,221,908,559, GFLOPs: 4.2219G
    model_surgery.get_flops(resnest.ResNest50())
    # >>>> FLOPs: 5,378,399,992, GFLOPs: 5.3784G
    ```
  - **[Deprecated] `tensorflow_addons`** is not imported by default. While reloading model depending on `GroupNormalization` like `MobileViTV2` from `h5` directly, needs to import `tensorflow_addons` manually first.
    ```py
    import tensorflow_addons as tfa

    model_path = os.path.expanduser('~/.keras/models/mobilevit_v2_050_256_imagenet.h5')
    mm = keras.models.load_model(model_path)
    ```
  - **Export TF model to onnx**. Needs `tf2onnx` for TF, `pip install onnx tf2onnx onnxsim onnxruntime`. For using PyTorch backend, exporting onnx is supported by PyTorch.
    ```py
    from keras_cv_attention_models import volo, nat, model_surgery
    mm = nat.DiNAT_Small(pretrained=True)
    model_surgery.export_onnx(mm, fuse_conv_bn=True, batch_size=1, simplify=True)
    # Exported simplified onnx: dinat_small.onnx

    # Run test
    from keras_cv_attention_models.imagenet import eval_func
    aa = eval_func.ONNXModelInterf(mm.name + '.onnx')
    inputs = np.random.uniform(size=[1, *mm.input_shape[1:]]).astype('float32')
    print(f"{np.allclose(aa(inputs), mm(inputs), atol=1e-5) = }")
    # np.allclose(aa(inputs), mm(inputs), atol=1e-5) = True
    ```
  - **Model summary** `model_summary.csv` contains gathered model info.
    - `params` for model params count in `M`
    - `flops` for FLOPs in `G`
    - `input` for model input shape
    - `acc_metrics` means `Imagenet Top1 Accuracy` for recognition models, `COCO val AP` for detection models
    - `inference` for `T4 inference query per second` with `batch_size=1 + trtexec`
    - `extra` means if any extra training info.
    ```py
    from keras_cv_attention_models import plot_func
    plot_series = ["efficientvit_b", "efficientvit_m", "efficientnet", "efficientnetv2"]
    plot_func.plot_model_summary(plot_series, x_label='inference', model_table="model_summary.csv")
    ```
    ![model_summary](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/33afa083-df15-46f1-b8a4-6a70851ba421)
  - **Code format** is using `line-length=160`:
    ```sh
    find ./* -name "*.py" | grep -v __init__ | xargs -I {} black -l 160 {}
    ```
## T4 Inference
  - **T4 Inference** in the model tables are tested using `trtexec` on `Tesla T4` with `CUDA=12.0.1-1, Driver=525.60.13`. All models are exported as ONNX using PyTorch backend, using `batch_szie=1` only. [Colab trtexec.ipynb](https://colab.research.google.com/drive/1xLwfvbZNqadkdAZu9b0UzOrETLo657oc?usp=drive_link).
  ```sh
  # Basic trtexec command
  trtexec --onnx=ConvFormerS18.onnx --fp16 --allowGPUFallback --useSpinWait # --useCudaGraph
  ```
## Layers
  - [attention_layers](keras_cv_attention_models/attention_layers) is `__init__.py` only, which imports core layers defined in model architectures. Like `RelativePositionalEmbedding` from `botnet`, `outlook_attention` from `volo`, and many other `Positional Embedding Layers` / `Attention Blocks`.
  ```py
  from keras_cv_attention_models import attention_layers
  aa = attention_layers.RelativePositionalEmbedding()
  print(f"{aa(tf.ones([1, 4, 14, 16, 256])).shape = }")
  # aa(tf.ones([1, 4, 14, 16, 256])).shape = TensorShape([1, 4, 14, 16, 14, 16])
  ```
## Model surgery
  - [model_surgery](keras_cv_attention_models/model_surgery) including functions used to change model parameters after built.
  ```py
  from keras_cv_attention_models import model_surgery
  mm = keras.applications.ResNet50()  # Trainable params: 25,583,592

  # Replace all ReLU with PReLU. Trainable params: 25,606,312
  mm = model_surgery.replace_ReLU(mm, target_activation='PReLU')

  # Fuse conv and batch_norm layers. Trainable params: 25,553,192
  mm = model_surgery.convert_to_fused_conv_bn_model(mm)
  ```
## ImageNet training and evaluating
  - [ImageNet](keras_cv_attention_models/imagenet) contains more detail usage and some comparing results.
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
  - [COCO](keras_cv_attention_models/coco) contains more detail usage.
  - `custom_dataset_script.py` can be used creating a `json` format file, which can be used as `--data_name xxx.json` for training, detail usage can be found in [Custom detection dataset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussioncomment-2460664).
  - Default parameters for `coco_train_script.py` is `EfficientDetD0` with `input_shape=(256, 256, 3), batch_size=64, mosaic_mix_prob=0.5, freeze_backbone_epochs=32, total_epochs=105`. Technically, it's any `pyramid structure backbone` + `EfficientDet / YOLOX header / YOLOR header` + `anchor_free / yolor / efficientdet anchors` combination supported.
  - Currently 4 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor", "yolov8"]`. Default `None` for `det_header` presets.
  - **NOTE: `YOLOV8` has a default `regression_len=64` for bbox output length. Typically it's `4` for other detection models, for yolov8 it's `reg_max=16 -> regression_len = 16 * 4 == 64`.**

    | anchors_mode | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------ | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free  | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor        | True              | 3           | None         | presets       | None       | offset=0.5      |
    | yolov8       | False             | 1           | 1            | [1]           | 1          | False           |

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
  - **`coco_eval_script.py`** is used for evaluating model AP / AR on COCO validation set. It has a dependency `pip install pycocotools` which is not in package requirements. More usage can be found in [COCO Evaluation](keras_cv_attention_models/coco#evaluation).
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
  - **[Experimental] Training using PyTorch backend**, currently using `ultralytics` dataset and validator process. The parameter `rect_val=False` means using fixed data shape `[640, 640]` for validator, or will by dynamic.
    ```py
    import os, sys
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append(os.path.expanduser("~/workspace/ultralytics/"))

    from keras_cv_attention_models.yolov8 import train, yolov8, torch_wrapper
    from keras_cv_attention_models import efficientnet

    # model Trainable params: 7,023,904, GFLOPs: 8.1815G
    bb = efficientnet.EfficientNetV2B0(input_shape=(3, 640, 640), num_classes=0)
    model = yolov8.YOLOV8_N(backbone=bb, classifier_activation=None, pretrained=None).cuda()
    # model = yolov8.YOLOV8_N(input_shape=(3, None, None), classifier_activation=None, pretrained=None).cuda()
    model = torch_wrapper.Detect(model)
    ema = train.train(model, dataset_path="coco.yaml", rect_val=False)
    ```
    ![yolov8_training](https://user-images.githubusercontent.com/5744524/235142289-cb6a4da0-1ea7-4261-afdd-03a3c36278b8.png)
## Visualizing
  - [Visualizing](keras_cv_attention_models/visualizing) is for visualizing convnet filters or attention map scores.
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
  - **Experimental** [Keras PyTorch Backend](keras_cv_attention_models/pytorch_backend).
  - **Set os environment `export KECAM_BACKEND='torch'` to enable this PyTorch backend.**
  - Currently supports most recognition and detection models except hornet / nfnets / volo. For detection models, using `torchvision.ops.nms` while running prediction.
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
    from keras_cv_attention_models.test_images import cat
    print(mm.decode_predictions(mm(mm.preprocess_input(cat())))[0])
    # [('n02124075', 'Egyptian_cat', 0.9597896), ('n02123045', 'tabby', 0.012809471), ...]
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
  - [Keras AotNet](keras_cv_attention_models/aotnet) is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer. Works like `byoanet` / `byobnet` from `timm`.
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
  - [Keras BEiT](keras_cv_attention_models/beit) includes models from [PDF 2106.08254 BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).

  | Model                 | Params  | FLOPs   | Input | Top1 Acc | T4 Inference |
  | --------------------- | ------- | ------- | ----- | -------- | ------------ |
  | [BeitBasePatch16, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_224_imagenet21k-ft1k.h5)  | 86.53M  | 17.61G  | 224   | 85.240   | 351.521 qps  |
  | - [384, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_384_imagenet21k-ft1k.h5)            | 86.74M  | 55.70G  | 384   | 86.808   | 150.047 qps  |
  | [BeitLargePatch16, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_224_imagenet21k-ft1k.h5) | 304.43M | 61.68G  | 224   | 87.476   | 99.8014 qps  |
  | - [384, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_384_imagenet21k-ft1k.h5)            | 305.00M | 191.65G | 384   | 88.382   | 48.0033 qps  |
  | - [512, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_512_imagenet21k-ft1k.h5)            | 305.67M | 363.46G | 512   | 88.584   | 23.8038 qps  |
## BEiTV2
  - [Keras BEiT](keras_cv_attention_models/beit) includes models from BeitV2 Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).

  | Model              | Params  | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------------ | ------- | ------ | ----- | -------- | ------------ |
  | BeitV2BasePatch16  | 86.53M  | 17.61G | 224   | 85.5     | 347.238 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_base_patch16_224_imagenet21k-ft1k.h5) | 86.53M  | 17.61G | 224   | 86.5     | 347.238 qps  |
  | BeitV2LargePatch16 | 304.43M | 61.68G | 224   | 87.3     | 98.1395 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_large_patch16_224_imagenet21k-ft1k.h5) | 304.43M | 61.68G | 224   | 88.4     | 98.1395 qps  |
## BotNet
  - [Keras BotNet](keras_cv_attention_models/botnet) is for [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ------ | ----- | -------- | ------------ |
  | BotNet50      | 21M    | 5.42G  | 224   |          | 747.134 qps  |
  | BotNet101     | 41M    | 9.13G  | 224   |          | 435.063 qps  |
  | BotNet152     | 56M    | 12.84G | 224   |          | 324.651 qps  |
  | [BotNet26T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet26t_256_imagenet.h5)     | 12.5M  | 3.30G  | 256   | 79.246   | 1183.29 qps  |
  | [BotNextECA26T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnext_eca26t_256_imagenet.h5) | 10.59M | 2.45G  | 256   | 79.270   | 1046.04 qps  |
  | [BotNetSE33T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet_se33t_256_imagenet.h5)   | 13.7M  | 3.89G  | 256   | 81.2     | 612.852 qps  |
## CAFormer
  - [Keras CAFormer](keras_cv_attention_models/caformer) is for [PDF 2210.13452 MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452.pdf). `CAFormer` is using 2 transformer stacks, while `ConvFormer` is all conv blocks.

  | Model                   | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ----------------------- | ------ | ----- | ----- | -------- | ------------ |
  | [CAFormerS18](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet.h5)             | 26M    | 4.1G  | 224   | 83.6     | 349.562 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet21k-ft1k.h5)      | 26M    | 4.1G  | 224   | 84.1     | 349.562 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet.h5)                   | 26M    | 13.4G | 384   | 85.0     | 171.51 qps   |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet21k-ft1k.h5) | 26M    | 13.4G | 384   | 85.4     | 171.51 qps   |
  | [CAFormerS36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet.h5)             | 39M    | 8.0G  | 224   | 84.5     | 189.586 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet21k-ft1k.h5)      | 39M    | 8.0G  | 224   | 85.8     | 189.586 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet.h5)                   | 39M    | 26.0G | 384   | 85.7     | 92.0302 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet21k-ft1k.h5) | 39M    | 26.0G | 384   | 86.9     | 92.0302 qps  |
  | [CAFormerM36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet.h5)             | 56M    | 13.2G | 224   | 85.2     | 139.914 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet21k-ft1k.h5)      | 56M    | 13.2G | 224   | 86.6     | 139.914 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet.h5)                   | 56M    | 42.0G | 384   | 86.2     | 62.9792 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet21k-ft1k.h5) | 56M    | 42.0G | 384   | 87.5     | 62.9792 qps  |
  | [CAFormerB36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet.h5)             | 99M    | 23.2G | 224   | 85.5     | 103.805 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet21k-ft1k.h5)      | 99M    | 23.2G | 224   | 87.4     | 103.805 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet.h5)                   | 99M    | 72.2G | 384   | 86.4     | 45.2969 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet21k-ft1k.h5) | 99M    | 72.2G | 384   | 88.1     | 45.2969 qps  |

  | Model                   | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ----------------------- | ------ | ----- | ----- | -------- | ------------ |
  | [ConvFormerS18](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet.h5)           | 27M    | 3.9G  | 224   | 83.0     | 314.063 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet21k-ft1k.h5)      | 27M    | 3.9G  | 224   | 83.7     | 314.063 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_384_imagenet.h5)                   | 27M    | 11.6G | 384   | 84.4     | 152.538 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) | 27M    | 11.6G | 384   | 85.0     | 152.538 qps  |
  | [ConvFormerS36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet.h5)           | 40M    | 7.6G  | 224   | 84.1     | 157.953 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet21k-ft1k.h5)      | 40M    | 7.6G  | 224   | 85.4     | 157.953 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet.h5)                   | 40M    | 22.4G | 384   | 85.4     | 81.961 qps   |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) | 40M    | 22.4G | 384   | 86.4     | 81.961 qps   |
  | [ConvFormerM36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet.h5)           | 57M    | 12.8G | 224   | 84.5     | 131.612 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet21k-ft1k.h5)      | 57M    | 12.8G | 224   | 86.1     | 131.612 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet.h5)                   | 57M    | 37.7G | 384   | 85.6     | 58.8572 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet21k-ft1k.h5) | 57M    | 37.7G | 384   | 86.9     | 58.8572 qps  |
  | [ConvFormerB36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet.h5)           | 100M   | 22.6G | 224   | 84.8     | 102.589 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet21k-ft1k.h5)      | 100M   | 22.6G | 224   | 87.0     | 102.589 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet.h5)                   | 100M   | 66.5G | 384   | 85.7     | 46.318 qps   |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet21k-ft1k.h5) | 100M   | 66.5G | 384   | 87.6     | 46.318 qps   |
## CMT
  - [Keras CMT](keras_cv_attention_models/cmt) is for [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf).

  | Model                              | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ---------------------------------- | ------ | ----- | ----- | -------- | ------------ |
  | CMTTiny, (Self trained 105 epochs) | 9.5M   | 0.65G | 160   | 77.4     | 342.318 qps  |
  | - [(305 epochs)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_160_imagenet.h5)                     | 9.5M   | 0.65G | 160   | 78.94    | 342.318 qps  |
  | - [224, (fine-tuned 69 epochs)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_224_imagenet.h5)      | 9.5M   | 1.32G | 224   | 80.73    | 293.799 qps  |
  | [CMTTiny_torch, (1000 epochs)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_torch_160_imagenet.h5)       | 9.5M   | 0.65G | 160   | 79.2     | 374.156 qps  |
  | [CMTXS_torch](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_xs_torch_192_imagenet.h5)                        | 15.2M  | 1.58G | 192   | 81.8     | 275.403 qps  |
  | [CMTSmall_torch](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_small_torch_224_imagenet.h5)                     | 25.1M  | 4.09G | 224   | 83.5     | 181.83 qps   |
  | [CMTBase_torch](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_base_torch_256_imagenet.h5)                      | 45.7M  | 9.42G | 256   | 84.5     | 108.942 qps  |
## CoaT
  - [Keras CoaT](keras_cv_attention_models/coat) is for [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).

  | Model         | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ----- | ----- | -------- | ------------ |
  | [CoaTLiteTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_tiny_imagenet.h5)  | 5.7M   | 1.60G | 224   | 77.5     | 485.178 qps  |
  | [CoaTLiteMini](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_mini_imagenet.h5)  | 11M    | 2.00G | 224   | 79.1     | 464.408 qps  |
  | [CoaTLiteSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_small_imagenet.h5) | 20M    | 3.97G | 224   | 81.9     | 264.449 qps  |
  | [CoaTTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_tiny_imagenet.h5)      | 5.5M   | 4.33G | 224   | 78.3     | 165.745 qps  |
  | [CoaTMini](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_mini_imagenet.h5)      | 10M    | 6.78G | 224   | 81.0     | 131.823 qps  |
## CoAtNet
  - [Keras CoAtNet](keras_cv_attention_models/coatnet) is for [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf).

  | Model                               | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ----------------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [CoAtNet0, (Self trained 105 epochs)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_160_imagenet.h5) | 23.3M  | 2.09G  | 160   | 80.48    | 425.88 qps   |
  | [CoAtNet0, (Self trained 305 epochs)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_224_imagenet.h5) | 23.8M  | 4.22G  | 224   | 82.79    | 425.88 qps   |
  | CoAtNet0                            | 25M    | 4.2G   | 224   | 81.6     | 425.88 qps   |
  | CoAtNet0, (Stride-2 DConv2D)        | 25M    | 4.6G   | 224   | 82.0     | 425.88 qps   |
  | CoAtNet1                            | 42M    | 8.4G   | 224   | 83.3     | 214.872 qps  |
  | CoAtNet1, (Stride-2 DConv2D)        | 42M    | 8.8G   | 224   | 83.5     | 214.872 qps  |
  | CoAtNet2                            | 75M    | 15.7G  | 224   | 84.1     | 162.664 qps  |
  | CoAtNet2, (Stride-2 DConv2D)        | 75M    | 16.6G  | 224   | 84.1     | 162.664 qps  |
  | CoAtNet2, ImageNet-21k              | 75M    | 16.6G  | 224   | 87.1     | 162.664 qps  |
  | CoAtNet3                            | 168M   | 34.7G  | 224   | 84.5     | 99.0514 qps  |
  | CoAtNet3, ImageNet-21k              | 168M   | 34.7G  | 224   | 87.6     | 99.0514 qps  |
  | CoAtNet3, ImageNet-21k              | 168M   | 203.1G | 512   | 87.9     | 99.0514 qps  |
  | CoAtNet4, ImageNet-21k              | 275M   | 360.9G | 512   | 88.1     | 57.4435 qps  |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 360.9G | 512   | 88.56    | 57.4435 qps  |
## ConvNeXt
  - [Keras ConvNeXt](keras_cv_attention_models/convnext) is for [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf).

  | Model                   | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ----------------------- | ------ | ------- | ----- | -------- | ------------ |
  | [ConvNeXtTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_imagenet.h5)            | 28M    | 4.49G   | 224   | 82.1     | 378.298 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_224_imagenet21k-ft1k.h5)      | 28M    | 4.49G   | 224   | 82.9     | 378.298 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_384_imagenet21k-ft1k.h5) | 28M    | 13.19G  | 384   | 84.1     | 186.891 qps  |
  | [ConvNeXtSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_imagenet.h5)           | 50M    | 8.73G   | 224   | 83.1     | 213.878 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_224_imagenet21k-ft1k.h5)      | 50M    | 8.73G   | 224   | 84.6     | 213.878 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_384_imagenet21k-ft1k.h5) | 50M    | 25.67G  | 384   | 85.8     | 108.964 qps  |
  | [ConvNeXtBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet.h5)            | 89M    | 15.42G  | 224   | 83.8     | 158.33 qps   |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet.h5)                   | 89M    | 45.32G  | 384   | 85.1     | 83.5147 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet21k-ft1k.h5)      | 89M    | 15.42G  | 224   | 85.8     | 158.33 qps   |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet21k-ft1k.h5) | 89M    | 45.32G  | 384   | 86.8     | 83.5147 qps  |
  | [ConvNeXtLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet.h5)           | 198M   | 34.46G  | 224   | 84.3     | 105.682 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet.h5)                   | 198M   | 101.28G | 384   | 85.5     | 48.3894 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet21k-ft1k.h5)      | 198M   | 34.46G  | 224   | 86.6     | 105.682 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet21k-ft1k.h5) | 198M   | 101.28G | 384   | 87.5     | 48.3894 qps  |
  | [ConvNeXtXlarge, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_224_imagenet21k-ft1k.h5)     | 350M   | 61.06G  | 224   | 87.0     | 70.8078 qps   |
  | - [384, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_384_imagenet21k-ft1k.h5)              | 350M   | 179.43G | 384   | 87.8     | 32.683 qps  |
  | [ConvNeXtXXLarge, clip](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xxlarge_clip-ft1k.h5)   | 846M   | 198.09G | 256   | 88.6     |              |
## ConvNeXtV2
  - [Keras ConvNeXt](keras_cv_attention_models/convnext) includes implementation of [PDF 2301.00808 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808.pdf). **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.

  | Model                   | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ----------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [ConvNeXtV2Atto](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_atto_imagenet.h5)          | 3.7M   | 0.55G  | 224   | 76.7     | 717.524 qps  |
  | [ConvNeXtV2Femto](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_femto_imagenet.h5)         | 5.2M   | 0.78G  | 224   | 78.5     | 744.726 qps  |
  | [ConvNeXtV2Pico](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_pico_imagenet.h5)          | 9.1M   | 1.37G  | 224   | 80.3     | 609.063 qps  |
  | [ConvNeXtV2Nano](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_imagenet.h5)          | 15.6M  | 2.45G  | 224   | 81.9     | 453.755 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_224_imagenet21k-ft1k.h5)      | 15.6M  | 2.45G  | 224   | 82.1     | 453.755 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_384_imagenet21k-ft1k.h5) | 15.6M  | 7.21G  | 384   | 83.4     | 221.756 qps  |
  | [ConvNeXtV2Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_imagenet.h5)          | 28.6M  | 4.47G  | 224   | 83.0     | 296.75 qps   |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_224_imagenet21k-ft1k.h5)      | 28.6M  | 4.47G  | 224   | 83.9     | 296.75 qps   |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_384_imagenet21k-ft1k.h5) | 28.6M  | 13.1G  | 384   | 85.1     | 145.222 qps  |
  | [ConvNeXtV2Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_imagenet.h5)          | 89M    | 15.4G  | 224   | 84.9     | 128.672 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_224_imagenet21k-ft1k.h5)      | 89M    | 15.4G  | 224   | 86.8     | 128.672 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_384_imagenet21k-ft1k.h5) | 89M    | 45.2G  | 384   | 87.7     | 66.0963 qps  |
  | [ConvNeXtV2Large](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_imagenet.h5)         | 198M   | 34.4G  | 224   | 85.8     | 87.7555 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_224_imagenet21k-ft1k.h5)      | 198M   | 34.4G  | 224   | 87.3     | 87.7555 qps  |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_384_imagenet21k-ft1k.h5) | 198M   | 101.1G | 384   | 88.2     | 37.6777 qps  |
  | [ConvNeXtV2Huge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_imagenet.h5)          | 660M   | 115G   | 224   | 86.3     |              |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_384_imagenet21k-ft1k.h5)      | 660M   | 337.9G | 384   | 88.7     |              |
  | - [ImageNet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_512_imagenet21k-ft1k.h5) | 660M   | 600.8G | 512   | 88.9     |              |
## CoTNet
  - [Keras CoTNet](keras_cv_attention_models/cotnet) is for [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------ |:------:| ------ | ----- |:--------:| ------------ |
  | [CotNet50](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224_imagenet.h5)     | 22.2M  | 3.25G  | 224   |   81.3   | 318.718 qps  |
  | [CotNetSE50D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se50d_224_imagenet.h5)  | 23.1M  | 4.05G  | 224   |   81.6   | 536.245 qps  |
  | [CotNet101](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224_imagenet.h5)    | 38.3M  | 6.07G  | 224   |   82.8   | 179.875 qps  |
  | [CotNetSE101D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se101d_224_imagenet.h5) | 40.9M  | 8.44G  | 224   |   83.2   | 258.401 qps  |
  | [CotNetSE152D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_224_imagenet.h5) | 55.8M  | 12.22G | 224   |   84.0   | 175.019 qps  |
  | [CotNetSE152D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_320_imagenet.h5) | 55.8M  | 24.92G | 320   |   84.6   | 175.019 qps  |
## DaViT
  - [Keras DaViT](keras_cv_attention_models/davit) is for [PDF 2204.03645 DaViT: Dual Attention Vision Transformers](https://arxiv.org/pdf/2204.03645.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ------ | ----- | -------- | ------------ |
  | [DaViT_T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_t_imagenet.h5)       | 28.36M | 4.56G  | 224   | 82.8     | 244.775 qps  |
  | [DaViT_S](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_s_imagenet.h5)       | 49.75M | 8.83G  | 224   | 84.2     | 149.76 qps   |
  | [DaViT_B](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_b_imagenet.h5)       | 87.95M | 15.55G | 224   | 84.6     | 105.758 qps  |
  | DaViT_L, 21k  | 196.8M | 103.2G | 384   | 87.5     | 34.1747 qps  |
  | DaViT_H, 1.5B | 348.9M | 327.3G | 512   | 90.2     | 13.0033 qps  |
  | DaViT_G, 1.5B | 1.406B | 1.022T | 512   | 90.4     |              |
## DiNAT
  - [Keras DiNAT](keras_cv_attention_models/nat) is for [PDF 2209.15001 Dilated Neighborhood Attention Transformer](https://arxiv.org/pdf/2209.15001.pdf).

  | Model                     | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [DiNAT_Mini](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_mini_imagenet.h5)                | 20.0M  | 2.73G  | 224   | 81.8     |              |
  | [DiNAT_Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_tiny_imagenet.h5)                | 27.9M  | 4.34G  | 224   | 82.7     |              |
  | [DiNAT_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_small_imagenet.h5)               | 50.7M  | 7.84G  | 224   | 83.8     |              |
  | [DiNAT_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_base_imagenet.h5)                | 89.8M  | 13.76G | 224   | 84.4     |              |
  | [DiNAT_Large, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_224_imagenet21k-ft1k.h5)          | 200.9M | 30.58G | 224   | 86.6     |              |
  | - [21k num_classes=21841](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_imagenet21k.h5)   | 200.9M | 30.58G | 224   |          |              |
  | - [22k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_384_imagenet21k-ft1k.h5)                | 200.9M | 89.86G | 384   | 87.4     |              |
  | [DiNAT_Large_K11, 22k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_k11_imagenet21k-ft1k.h5) | 201.1M | 92.57G | 384   | 87.5     |              |
## DINOv2
  - [Keras DINOv2](keras_cv_attention_models/beit) includes models from [PDF 2304.07193 DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf).

  | Model              | Params  | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------------------ | ------- | ------- | ----- | -------- | ------------ |
  | [DINOv2_ViT_Small14](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_small14_518_imagenet.h5) | 22.83M  | 47.23G  | 518   | 81.1     | 158.769 qps  |
  | [DINOv2_ViT_Base14](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_base14_518_imagenet.h5)  | 88.12M  | 152.6G  | 518   | 84.5     | 54.2718 qps  |
  | [DINOv2_ViT_Large14](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_large14_518_imagenet.h5) | 306.4M  | 509.6G  | 518   | 86.3     | 15.9247 qps  |
  | [DINOv2_ViT_Giant14](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_giant14_518_imagenet.h5) | 1139.6M | 1790.3G | 518   | 86.5     |              |
## EdgeNeXt
  - [Keras EdgeNeXt](keras_cv_attention_models/edgenext) is for [PDF 2206.10589 EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/pdf/2206.10589.pdf).

  | Model             | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ----------------- | ------ | ------ | ----- | -------- | ------------ |
  | [EdgeNeXt_XX_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_xx_small_256_imagenet.h5) | 1.33M  | 266M   | 256   | 71.23    | 954.051 qps  |
  | [EdgeNeXt_X_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_x_small_256_imagenet.h5)  | 2.34M  | 547M   | 256   | 74.96    | 677.654 qps  |
  | [EdgeNeXt_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_imagenet.h5)    | 5.59M  | 1.27G  | 256   | 79.41    | 578.389 qps  |
  | - [usi](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_usi.h5)             | 5.59M  | 1.27G  | 256   | 81.07    | 578.389 qps  |
## EfficientFormer
  - [Keras EfficientFormer](keras_cv_attention_models/efficientformer) is for [PDF 2206.01191 EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191.pdf).

  | Model                      | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | -------------------------- | ------ | ----- | ----- | -------- | ------------ |
  | [EfficientFormerL1, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l1_224_imagenet.h5) | 12.3M  | 1.31G | 224   | 79.2     | 1226.31 qps  |
  | [EfficientFormerL3, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l3_224_imagenet.h5) | 31.4M  | 3.95G | 224   | 82.4     | 530.625 qps  |
  | [EfficientFormerL7, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l7_224_imagenet.h5) | 74.4M  | 9.79G | 224   | 83.3     | 217.117 qps  |
## EfficientFormerV2
  - [Keras EfficientFormer](keras_cv_attention_models/efficientformer) includes implementation of [PDF 2212.08059 Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/pdf/2212.08059.pdf).

  | Model                        | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ---------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [EfficientFormerV2S0, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s0_224_imagenet.h5) | 3.60M  | 405.2M | 224   | 76.2     | 1239.8 qps   |
  | [EfficientFormerV2S1, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s1_224_imagenet.h5) | 6.19M  | 665.6M | 224   | 79.7     | 964.7 qps    |
  | [EfficientFormerV2S2, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s2_224_imagenet.h5) | 12.7M  | 1.27G  | 224   | 82.0     | 585.585 qps  |
  | [EfficientFormerV2L, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_l_224_imagenet.h5)  | 26.3M  | 2.59G  | 224   | 83.5     | 392.825 qps  |
## EfficientNet
  - [Keras EfficientNet](keras_cv_attention_models/efficientnet) includes implementation of [PDF 1911.04252 Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf).

  | V1 Model                       | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------------------------------ | ------ | ------- | ----- | -------- | ------------ |
  | [EfficientNetV1B0](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-imagenet.h5)               | 5.3M   | 0.39G   | 224   | 77.6     | 1189.34 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b0-noisy_student.h5)                 | 5.3M   | 0.39G   | 224   | 78.8     | 1189.34 qps  |
  | [EfficientNetV1B1](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-imagenet.h5)               | 7.8M   | 0.70G   | 240   | 79.6     | 790.774 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b1-noisy_student.h5)                 | 7.8M   | 0.70G   | 240   | 81.5     | 790.774 qps  |
  | [EfficientNetV1B2](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-imagenet.h5)               | 9.1M   | 1.01G   | 260   | 80.5     | 699.629 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b2-noisy_student.h5)                 | 9.1M   | 1.01G   | 260   | 82.4     | 699.629 qps  |
  | [EfficientNetV1B3](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-imagenet.h5)               | 12.2M  | 1.86G   | 300   | 81.9     | 503.373 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b3-noisy_student.h5)                 | 12.2M  | 1.86G   | 300   | 84.1     | 503.373 qps  |
  | [EfficientNetV1B4](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-imagenet.h5)               | 19.3M  | 4.46G   | 380   | 83.3     | 281.455 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b4-noisy_student.h5)                 | 19.3M  | 4.46G   | 380   | 85.3     | 281.455 qps  |
  | [EfficientNetV1B5](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-imagenet.h5)               | 30.4M  | 10.40G  | 456   | 84.3     | 153.726 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b5-noisy_student.h5)                 | 30.4M  | 10.40G  | 456   | 86.1     | 153.726 qps  |
  | [EfficientNetV1B6](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-imagenet.h5)               | 43.0M  | 19.29G  | 528   | 84.8     | 92.0942 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b6-noisy_student.h5)                 | 43.0M  | 19.29G  | 528   | 86.4     | 92.0942 qps  |
  | [EfficientNetV1B7](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-imagenet.h5)               | 66.3M  | 38.13G  | 600   | 85.2     | 54.9365 qps  |
  | - [NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-b7-noisy_student.h5)                 | 66.3M  | 38.13G  | 600   | 86.9     | 54.9365 qps  |
  | [EfficientNetV1L2, NoisyStudent](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnetv1-l2-noisy_student.h5) | 480.3M | 477.98G | 800   | 88.4     | 8.35729 qps  |
## EfficientNetV2
  - [Keras EfficientNet](keras_cv_attention_models/efficientnet) includes implementation of [PDF 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

  | V2 Model                   | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [EfficientNetV2B0](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-imagenet.h5)           | 7.1M   | 0.72G  | 224   | 78.7     | 1180.15 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b0-21k-ft1k.h5)         | 7.1M   | 0.72G  | 224   | 77.55?   | 1180.15 qps  |
  | [EfficientNetV2B1](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-imagenet.h5)           | 8.1M   | 1.21G  | 240   | 79.8     | 958.012 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b1-21k-ft1k.h5)         | 8.1M   | 1.21G  | 240   | 79.03?   | 958.012 qps  |
  | [EfficientNetV2B2](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-imagenet.h5)           | 10.1M  | 1.71G  | 260   | 80.5     | 829.3 qps    |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b2-21k-ft1k.h5)         | 10.1M  | 1.71G  | 260   | 79.48?   | 829.3 qps    |
  | [EfficientNetV2B3](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-imagenet.h5)           | 14.4M  | 3.03G  | 300   | 82.1     | 580.35 qps   |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-b3-21k-ft1k.h5)         | 14.4M  | 3.03G  | 300   | 82.46?   | 580.35 qps   |
  | [EfficientNetV2T](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-t-imagenet.h5)            | 13.6M  | 3.18G  | 288   | 82.34    | 532.972 qps  |
  | [EfficientNetV2T_GC](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-t-gc-imagenet.h5)         | 13.7M  | 3.19G  | 288   | 82.46    | 383.378 qps  |
  | [EfficientNetV2S](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-imagenet.h5)            | 21.5M  | 8.41G  | 384   | 83.9     | 361.509 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-21k-ft1k.h5)         | 21.5M  | 8.41G  | 384   | 84.9     | 361.509 qps  |
  | [EfficientNetV2M](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-imagenet.h5)            | 54.1M  | 24.69G | 480   | 85.2     | 157.764 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-m-21k-ft1k.h5)         | 54.1M  | 24.69G | 480   | 86.2     | 157.764 qps  |
  | [EfficientNetV2L](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-imagenet.h5)            | 119.5M | 56.27G | 480   | 85.7     | 90.4257 qps  |
  | - [ImageNet21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-l-21k-ft1k.h5)         | 119.5M | 56.27G | 480   | 86.9     | 90.4257 qps  |
  | [EfficientNetV2XL, 21k-ft1k](https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-xl-21k-ft1k.h5) | 206.8M | 93.66G | 512   | 87.2     | 58.7622 qps  |
## EfficientViT_B
  - [Keras EfficientViT_B](keras_cv_attention_models/efficientvit) is for Paper [PDF 2205.14756 EfficientViT: Lightweight Multi-Scale Attention for On-Device Semantic Segmentation](https://arxiv.org/pdf/2205.14756.pdf).

  | Model           | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | --------------- | ------ | ----- | ----- | -------- | ------------ |
  | [EfficientViT_B1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_224_imagenet.h5) | 9.10M  | 0.58G | 224   | 79.4     | 1052.8 qps   |
  | - [256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_256_imagenet.h5)           | 9.10M  | 0.78G | 256   | 79.9     | 903.761 qps  |
  | - [288](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_288_imagenet.h5)            | 9.10M  | 1.03G | 288   | 80.4     | 739.044 qps  |
  | [EfficientViT_B2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_224_imagenet.h5) | 24.33M | 1.68G | 224   | 82.1     | 620.375 qps  |
  | - [256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_256_imagenet.h5)            | 24.33M | 2.25G | 256   | 82.7     | 538.368 qps  |
  | - [288](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_288_imagenet.h5)            | 24.33M | 2.92G | 288   | 83.1     | 444.323 qps  |
  | [EfficientViT_B3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_224_imagenet.h5) | 48.65M | 4.14G | 224   | 83.5     | 351.024 qps  |
  | - [256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_256_imagenet.h5)            | 48.65M | 5.51G | 256   | 83.8     | 304.131 qps  |
  | - [288](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_288_imagenet.h5)            | 48.65M | 7.14G | 288   | 84.2     | 230.509 qps  |
## EfficientViT_M
  - [Keras EfficientViT_M](keras_cv_attention_models/efficientvit) is for Paper [PDF 2305.07027 EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/pdf/2305.07027.pdf).

  | Model           | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | --------------- | ------ | ----- | ----- | -------- | ------------ |
  | [EfficientViT_M0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m0_224_imagenet.h5) | 2.35M  | 79.4M | 224   | 63.2     | 873.905 qps  |
  | [EfficientViT_M1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m1_224_imagenet.h5) | 2.98M  | 167M  | 224   | 68.4     | 1003.03 qps  |
  | [EfficientViT_M2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m2_224_imagenet.h5) | 4.19M  | 201M  | 224   | 70.8     | 995.149 qps  |
  | [EfficientViT_M3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m3_224_imagenet.h5) | 6.90M  | 263M  | 224   | 73.4     | 848.056 qps  |
  | [EfficientViT_M4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m4_224_imagenet.h5) | 8.80M  | 299M  | 224   | 74.3     | 761.256 qps  |
  | [EfficientViT_M5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m5_224_imagenet.h5) | 12.47M | 522M  | 224   | 77.1     | 630.102 qps  |
## EVA
  - [Keras EVA](keras_cv_attention_models/beit) includes models from [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf).

  | Model                 | Params  | FLOPs    | Input | Top1 Acc | T4 Inference |
  | --------------------- | ------- | -------- | ----- | -------- | ------------ |
  | [EvaLargePatch14, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_196_imagenet21k-ft1k.h5)  | 304.14M | 61.65G   | 196   | 88.59    | 101.135 qps  |
  | - [336, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_336_imagenet21k-ft1k.h5)            | 304.53M | 191.55G  | 336   | 89.20    | 45.2344 qps  |
  | [EvaGiantPatch14, clip](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_224_imagenet21k-ft1k.h5) | 1012.6M | 267.40G  | 224   | 89.10    |              |
  | - [m30m](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_336_imagenet21k-ft1k.h5)                | 1013.0M | 621.45G  | 336   | 89.57    |              |
  | - [m30m](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_560_imagenet21k-ft1k.h5)                | 1014.4M | 1911.61G | 560   | 89.80    |              |
## EVA02
  - [Keras EVA02](keras_cv_attention_models/beit) includes models from [PDF 2303.11331 EVA: EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/pdf/2303.11331.pdf).

  | Model                                  | Params  | FLOPs   | Input | Top1 Acc | T4 Inference |
  | -------------------------------------- | ------- | ------- | ----- | -------- | ------------ |
  | [EVA02TinyPatch14, mim_in22k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_tiny_patch14_336_mim_in22k_ft1k.h5)       | 5.76M   | 4.72G   | 336   | 80.658   | 317.805 qps  |
  | [EVA02SmallPatch14, mim_in22k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_small_patch14_336_mim_in22k_ft1k.h5)      | 22.13M  | 15.57G  | 336   | 85.74    | 176.416 qps  |
  | [EVA02BasePatch14, mim_in22k_ft22k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_base_patch14_448_mim_in22k_ft22k_ft1k.h5) | 87.12M  | 107.6G  | 448   | 88.692   | 34.7612 qps  |
  | [EVA02LargePatch14, mim_m38m_ft22k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_large_patch14_448_mim_m38m_ft22k_ft1k.h5) | 305.08M | 363.68G | 448   | 90.054   | 11.6624 qps  |
## FasterNet
  - [Keras FasterNet](keras_cv_attention_models/fasternet) includes implementation of [PDF 2303.03667 Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks ](https://arxiv.org/pdf/2303.03667.pdf).

  | Model       | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ----------- | ------ | ------ | ----- | -------- | ------------ |
  | [FasterNetT0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t0_imagenet.h5) | 3.9M   | 0.34G  | 224   | 71.9     | 1775.67 qps  |
  | [FasterNetT1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t1_imagenet.h5) | 7.6M   | 0.85G  | 224   | 76.2     | 1703.46 qps  |
  | [FasterNetT2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t2_imagenet.h5) | 15.0M  | 1.90G  | 224   | 78.9     | 1343.53 qps  |
  | [FasterNetS](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_s_imagenet.h5)  | 31.1M  | 4.55G  | 224   | 81.3     | 826.201 qps  |
  | [FasterNetM](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_m_imagenet.h5)  | 53.5M  | 8.72G  | 224   | 83.0     | 436.996 qps  |
  | [FasterNetL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_l_imagenet.h5)  | 93.4M  | 15.49G | 224   | 83.5     | 335.379 qps  |
## FBNetV3
  - [Keras FBNetV3](keras_cv_attention_models/mobilenetv3_family#fbnetv3) includes implementation of [PDF 2006.02049 FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining](https://arxiv.org/pdf/2006.02049.pdf).

  | Model    | Params | FLOPs    | Input | Top1 Acc | T4 Inference |
  | -------- | ------ | -------- | ----- | -------- | ------------ |
  | [FBNetV3B](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_b_imagenet.h5) | 5.57M  | 539.82M  | 256   | 79.15    | 784.788 qps  |
  | [FBNetV3D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_d_imagenet.h5) | 10.31M | 665.02M  | 256   | 79.68    | 753.521 qps  |
  | [FBNetV3G](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_g_imagenet.h5) | 16.62M | 1379.30M | 256   | 82.05    | 531.545 qps  |
## FlexiViT
  - [Keras FlexiViT](keras_cv_attention_models/beit) includes models from [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).

  | Model         | Params  | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------- | ------- | ------ | ----- | -------- | ------------ |
  | [FlexiViTSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_small_240_imagenet.h5) | 22.06M  | 5.36G  | 240   | 82.53    | 743.069 qps  |
  | [FlexiViTBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_base_240_imagenet.h5)  | 86.59M  | 20.33G | 240   | 84.66    | 323.861 qps  |
  | [FlexiViTLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_large_240_imagenet.h5) | 304.47M | 71.09G | 240   | 85.64    | 93.8481 qps  |
## GCViT
  - [Keras GCViT](keras_cv_attention_models/gcvit) includes implementation of [PDF 2206.09959 Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf).

  | Model        | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ------------ | ------ | ----- | ----- | -------- | ------------ |
  | [GCViT_XXTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_xx_tiny_224_imagenet.h5) | 12.0M  | 2.15G | 224   | 79.8     | 345.076 qps  |
  | [GCViT_XTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_x_tiny_224_imagenet.h5)  | 20.0M  | 2.96G | 224   | 82.04    | 258.664 qps  |
  | [GCViT_Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_tiny_224_imagenet.h5)   | 28.2M  | 4.83G | 224   | 83.4     | 175.8 qps    |
  | [GCViT_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_small_224_imagenet.h5)  | 51.1M  | 8.63G | 224   | 83.95    | 135.64 qps   |
  | [GCViT_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_base_224_imagenet.h5)   | 90.3M  | 14.9G | 224   | 84.47    | 109.29 qps   |
## GhostNet
  - [Keras GhostNet](keras_cv_attention_models/ghostnet) includes implementation of [PDF 1911.11907 GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------ | ------ | ------ | ----- | -------- | ------------ |
  | [GhostNet_050](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_050_imagenet.h5) | 2.59M  | 42.6M  | 224   | 66.88    | 1423.53 qps  |
  | [GhostNet_100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_100_imagenet.h5) | 5.18M  | 141.7M | 224   | 74.16    | 1332.75 qps  |
  | [GhostNet_130](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_imagenet.h5) | 7.36M  | 227.7M | 224   | 75.79    | 1195.01 qps  |
  | - [ssld](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_ssld.h5)       | 7.36M  | 227.7M | 224   | 79.38    | 1195.01 qps  |
## GhostNetV2
  - [Keras GhostNet](keras_cv_attention_models/ghostnet) includes implementation of [PDF GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf).

  | Model             | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ----------------- | ------ | ------ | ----- | -------- | ------------ |
  | [GhostNetV2_100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnetv2_100_imagenet.h5)    | 6.12M  | 168.5M | 224   | 74.41    | 928.209 qps  |
  | GhostNetV2 (1.0x) | 6.12M  | 168.5M | 224   | 75.3     |              |
  | GhostNetV2 (1.3x) | 8.96M  | 271.1M | 224   | 76.9     |              |
  | GhostNetV2 (1.6x) | 12.39M | 400.9M | 224   | 77.8     |              |
## GMLP
  - [Keras GMLP](keras_cv_attention_models/mlp_family#gmlp) includes implementation of [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).

  | Model      | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ---------- | ------ | ------ | ----- | -------- | ------------ |
  | GMLPTiny16 | 6M     | 1.35G  | 224   | 72.3     | 235.083 qps  |
  | [GMLPS16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/gmlp_s16_imagenet.h5)    | 20M    | 4.44G  | 224   | 79.6     | 133.468 qps  |
  | GMLPB16    | 73M    | 15.82G | 224   | 81.6     | 73.6252 qps  |
## GPViT
  - [Keras GPViT](keras_cv_attention_models/gpvit) includes implementation of [PDF 2212.06795 GPVIT: A HIGH RESOLUTION NON-HIERARCHICAL VISION TRANSFORMER WITH GROUP PROPAGATION](https://arxiv.org/pdf/2212.06795.pdf).

  | Model    | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------- | ------ | ------ | ----- | -------- | ------------ |
  | [GPViT_L1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l1_224_imagenet.h5) | 9.59M  | 6.15G  | 224   | 80.5     | 215.89 qps   |
  | [GPViT_L2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l2_224_imagenet.h5) | 24.2M  | 15.74G | 224   | 83.4     | 140.915 qps  |
  | [GPViT_L3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l3_224_imagenet.h5) | 36.7M  | 23.54G | 224   | 84.1     | 132.775 qps  |
  | [GPViT_L4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l4_224_imagenet.h5) | 75.5M  | 48.29G | 224   | 84.3     | 94.0427 qps  |
## HaloNet
  - [Keras HaloNet](keras_cv_attention_models/halonet) is for [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).

  | Model          | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | -------------- | ------ | ------- | ----- | -------- | ------------ |
  | [HaloNextECA26T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonext_eca26t_256_imagenet.h5) | 10.7M  | 2.43G   | 256   | 79.50    | 1011.11 qps  |
  | [HaloNet26T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet26t_256_imagenet.h5)     | 12.5M  | 3.18G   | 256   | 79.13    | 1056.81 qps  |
  | [HaloNetSE33T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet_se33t_256_imagenet.h5)   | 13.7M  | 3.55G   | 256   | 80.99    | 591.666 qps  |
  | [HaloRegNetZB](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/haloregnetz_b_224_imagenet.h5)   | 11.68M | 1.97G   | 224   | 81.042   | 582.876 qps  |
  | [HaloNet50T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halonet50t_256_imagenet.h5)     | 22.7M  | 5.29G   | 256   | 81.70    | 509.481 qps  |
  | [HaloBotNet50T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/halonet/halobotnet50t_256_imagenet.h5)  | 22.6M  | 5.02G   | 256   | 82.0     | 435.006 qps  |
## HorNet
  - [Keras HorNet](keras_cv_attention_models/hornet) is for [PDF 2207.14284 HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/pdf/2207.14284.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ------ | ----- | -------- | ------------ |
  | [HorNetTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_224_imagenet.h5)    | 22.4M  | 4.01G  | 224   | 82.8     | 176.457 qps  |
  | [HorNetTinyGF](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_gf_224_imagenet.h5)  | 23.0M  | 3.94G  | 224   | 83.0     |              |
  | [HorNetSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_224_imagenet.h5)   | 49.5M  | 8.87G  | 224   | 83.8     | 132.895 qps  |
  | [HorNetSmallGF](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_gf_224_imagenet.h5) | 50.4M  | 8.77G  | 224   | 84.0     |              |
  | [HorNetBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_224_imagenet.h5)    | 87.3M  | 15.65G | 224   | 84.2     | 105.358 qps  |
  | [HorNetBaseGF](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_gf_224_imagenet.h5)  | 88.4M  | 15.51G | 224   | 84.3     |              |
  | [HorNetLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_224_imagenet22k.h5)   | 194.5M | 34.91G | 224   | 86.8     | 71.6244 qps  |
  | [HorNetLargeGF](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_224_imagenet22k.h5) | 196.3M | 34.72G | 224   | 87.0     |              |
  | [HorNetLargeGF](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_384_imagenet22k.h5) | 201.8M | 102.0G | 384   | 87.7     |              |
## IFormer
  - [Keras IFormer](keras_cv_attention_models/iformer) is for [PDF 2205.12956 Inception Transformer](https://arxiv.org/pdf/2205.12956.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------ | ------ | ------ | ----- | -------- | ------------ |
  | [IFormerSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_224_imagenet.h5) | 19.9M  | 4.88G  | 224   | 83.4     | 250 qps      |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_384_imagenet.h5)        | 20.9M  | 16.29G | 384   | 84.6     | 130.196 qps  |
  | [IFormerBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_224_imagenet.h5)  | 47.9M  | 9.44G  | 224   | 84.6     | 147.068 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_384_imagenet.h5)        | 48.9M  | 30.86G | 384   | 85.7     | 78.2124 qps  |
  | [IFormerLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_224_imagenet.h5) | 86.6M  | 14.12G | 224   | 84.6     | 112.554 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_384_imagenet.h5)        | 87.7M  | 45.74G | 384   | 85.8     | 62.0674 qps  |
## InceptionNeXt
  - [Keras InceptionNeXt](keras_cv_attention_models/inceptionnext) is for [PDF 2303.16900 InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/pdf/2303.16900.pdf).

  | Model              | Params | FLOP s | Input | Top1 Acc | T4 Inference |
  | ------------------ | ------ | ------ | ----- | -------- | ------------ |
  | [InceptionNeXtTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_tiny_imagenet.h5)  | 28.05M | 4.21G  | 224   | 82.3     | 637.253 qps  |
  | [InceptionNeXtSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_small_imagenet.h5) | 49.37M | 8.39G  | 224   | 83.5     | 346.174 qps  |
  | [InceptionNeXtBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_224_imagenet.h5)  | 86.67M | 14.88G | 224   | 84.0     | 269.618 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_384_imagenet.h5)              | 86.67M | 43.73G | 384   | 85.2     | 151.412 qps  |
## LCNet
  - [Keras LCNet](keras_cv_attention_models/mobilenetv3_family#lcnet) includes implementation of [PDF 2109.15099 PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf).

  | Model    | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | -------- | ------ | ------- | ----- | -------- | ------------ |
  | [LCNet050](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_050_imagenet.h5) | 1.88M  | 46.02M  | 224   | 63.10    | 3229.05 qps  |
  | - [ssld](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_050_ssld.h5)   | 1.88M  | 46.02M  | 224   | 66.10    | 3229.05 qps  |
  | [LCNet075](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_075_imagenet.h5) | 2.36M  | 96.82M  | 224   | 68.82    | 2690.82 qps  |
  | [LCNet100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_100_imagenet.h5) | 2.95M  | 158.28M | 224   | 72.10    | 2343.16 qps  |
  | - [ssld](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_100_ssld.h5)   | 2.95M  | 158.28M | 224   | 74.39    | 2343.16 qps  |
  | [LCNet150](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_150_imagenet.h5) | 4.52M  | 338.05M | 224   | 73.71    | 2443.05 qps  |
  | [LCNet200](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_200_imagenet.h5) | 6.54M  | 585.35M | 224   | 75.18    | 2147.76 qps  |
  | [LCNet250](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_250_imagenet.h5) | 9.04M  | 900.16M | 224   | 76.60    | 1789.1 qps   |
  | - [ssld](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_250_ssld.h5)   | 9.04M  | 900.16M | 224   | 80.82    | 1789.1 qps   |
## LeViT
  - [Keras LeViT](keras_cv_attention_models/levit) is for [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).

  | Model              | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ------------------ | ------ | ----- | ----- | -------- | ------------ |
  | [LeViT128S, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128s_imagenet.h5) | 7.8M   | 0.31G | 224   | 76.6     | 865.62 qps   |
  | [LeViT128, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128_imagenet.h5)  | 9.2M   | 0.41G | 224   | 78.6     | 672.763 qps  |
  | [LeViT192, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit192_imagenet.h5)  | 11M    | 0.66G | 224   | 80.0     | 636.368 qps  |
  | [LeViT256, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit256_imagenet.h5)  | 19M    | 1.13G | 224   | 81.6     | 582.359 qps  |
  | [LeViT384, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit384_imagenet.h5)  | 39M    | 2.36G | 224   | 82.6     | 455.795 qps  |
## MaxViT
  - [Keras MaxViT](keras_cv_attention_models/maxvit) is for [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).

  | Model                      | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [MaxViT_Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_224_imagenet.h5)                | 31M    | 5.6G   | 224   | 83.62    | 201.674 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_384_imagenet.h5)                      | 31M    | 17.7G  | 384   | 85.24    | 94.0024 qps  |
  | - [512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_512_imagenet.h5)                      | 31M    | 33.7G  | 512   | 85.72    | 53.0237 qps  |
  | [MaxViT_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_224_imagenet.h5)               | 69M    | 11.7G  | 224   | 84.45    | 150.903 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_384_imagenet.h5)                    | 69M    | 36.1G  | 384   | 85.74    | 62.5982 qps  |
  | - [512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_512_imagenet.h5)                    | 69M    | 67.6G  | 512   | 86.19    | 34.3803 qps  |
  | [MaxViT_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet.h5)                | 119M   | 24.2G  | 224   | 84.95    | 76.544 qps   |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet.h5)                      | 119M   | 74.2G  | 384   | 86.34    | 32.3341 qps   |
  | - [512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet.h5)                      | 119M   | 138.5G | 512   | 86.66    | 18.3942 qps   |
  | - [imagenet21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet21k.h5)              | 135M   | 24.2G  | 224   |          | 76.544 qps   |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet21k-ft1k.h5)     | 119M   | 74.2G  | 384   | 88.24    | 32.3341 qps   |
  | - [imagenet21k-ft1k, 512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet21k-ft1k.h5)     | 119M   | 138.5G | 512   | 88.38    | 18.3942 qps   |
  | [MaxViT_Large](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet.h5)               | 212M   | 43.9G  | 224   | 85.17    | 59.1861 qps  |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet.h5)                    | 212M   | 133.1G | 384   | 86.40    | 24.3532 qps  |
  | - [512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet.h5)                    | 212M   | 245.4G | 512   | 86.70    | 13.3465 qps  |
  | - [imagenet21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet21k.h5)              | 233M   | 43.9G  | 224   |          | 59.1861 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet21k-ft1k.h5)     | 212M   | 133.1G | 384   | 88.32    |24.3532 qps  |
  | - [imagenet21k-ft1k, 512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet21k-ft1k.h5)     | 212M   | 245.4G | 512   | 88.46    | 13.3465 qps  |
  | [MaxViT_XLarge, imagenet21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_224_imagenet21k.h5) | 507M   | 97.7G  | 224   |          | 38.8319 qps  |
  | - [imagenet21k-ft1k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_384_imagenet21k-ft1k.h5)    | 475M   | 293.7G | 384   | 88.51    | 14.2965 qps  |
  | - [imagenet21k-ft1k, 512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_512_imagenet21k-ft1k.h5)    | 475M   | 535.2G | 512   | 88.70    | 8.06868 qps  |
## MLP mixer
  - [Keras MLP mixer](keras_cv_attention_models/mlp_family#mlp-mixer) includes implementation of [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).

  | Model            | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ---------------- | ------ | ------- | ----- | -------- | ------------ |
  | MLPMixerS32, JFT | 19.1M  | 1.01G   | 224   | 68.70    | 500.665 qps  |
  | MLPMixerS16, JFT | 18.5M  | 3.79G   | 224   | 73.83    | 438.837 qps  |
  | MLPMixerB32, JFT | 60.3M  | 3.25G   | 224   | 75.53    | 248 qps      |
  | - [imagenet_sam](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b32_imagenet_sam.h5)   | 60.3M  | 3.25G   | 224   | 72.47    | 248 qps      |
  | [MLPMixerB16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet.h5)      | 59.9M  | 12.64G  | 224   | 76.44    | 206.581 qps  |
  | - [imagenet21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet21k.h5)    | 59.9M  | 12.64G  | 224   | 80.64    | 206.581 qps  |
  | - [imagenet_sam](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet_sam.h5)   | 59.9M  | 12.64G  | 224   | 77.36    | 206.581 qps  |
  | - JFT            | 59.9M  | 12.64G  | 224   | 80.00    | 206.581 qps  |
  | MLPMixerL32, JFT | 206.9M | 11.30G  | 224   | 80.67    | 95.8038 qps  |
  | [MLPMixerL16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet.h5)      | 208.2M | 44.66G  | 224   | 71.76    | 77.4111 qps  |
  | - [imagenet21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet21k.h5)    | 208.2M | 44.66G  | 224   | 82.89    | 77.4111 qps  |
  | - JFT            | 208.2M | 44.66G  | 224   | 84.82    | 77.4111 qps  |
  | - 448            | 208.2M | 178.54G | 448   | 83.91    |              |
  | - 448, JFT       | 208.2M | 178.54G | 448   | 86.78    |              |
  | MLPMixerH14, JFT | 432.3M | 121.22G | 224   | 86.32    | 44.4475 qps  |
  | - 448, JFT       | 432.3M | 484.73G | 448   | 87.94    |              |
## MobileNetV3
  - [Keras MobileNetV3](keras_cv_attention_models/mobilenetv3_family#mobilenetv3) includes implementation of [PDF 1905.02244 Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

  | Model               | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------------------- | ------ | ------- | ----- | -------- | ------------ |
  | [MobileNetV3Small050](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_050_imagenet.h5) | 1.29M  | 24.92M  | 224   | 57.89    | 2599.34 qps  |
  | [MobileNetV3Small075](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_075_imagenet.h5) | 2.04M  | 44.35M  | 224   | 65.24    | 2416.64 qps  |
  | [MobileNetV3Small100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_100_imagenet.h5) | 2.54M  | 57.62M  | 224   | 67.66    | 2241.26 qps  |
  | [MobileNetV3Large075](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_075_imagenet.h5) | 3.99M  | 156.30M | 224   | 73.44    | 1911.1 qps   |
  | [MobileNetV3Large100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_imagenet.h5) | 5.48M  | 218.73M | 224   | 75.77    | 1765.26 qps  |
  | - [miil](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_mill.h5)              | 5.48M  | 218.73M | 224   | 77.92    | 1765.26 qps  |
## MobileViT
  - [Keras MobileViT](keras_cv_attention_models/mobilevit) is for [PDF 2110.02178 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER](https://arxiv.org/pdf/2110.02178.pdf).

  | Model         | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ----- | ----- | -------- | ------------ |
  | [MobileViT_XXS](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xxs_imagenet.h5) | 1.3M   | 0.42G | 256   | 69.0     | 1215.21 qps  |
  | [MobileViT_XS](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xs_imagenet.h5)  | 2.3M   | 1.05G | 256   | 74.7     | 877.311 qps  |
  | [MobileViT_S](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_s_imagenet.h5)   | 5.6M   | 2.03G | 256   | 78.3     | 597.887 qps  |
## MobileViT_V2
  - [Keras MobileViT_V2](keras_cv_attention_models/mobilevit) is for [PDF 2206.02680 Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/pdf/2206.02680.pdf).

  | Model              | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | ------------------ | ------ | ----- | ----- | -------- | ------------ |
  | [MobileViT_V2_050](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_050_256_imagenet.h5)   | 1.37M  | 0.47G | 256   | 70.18    | 847.04 qps   |
  | [MobileViT_V2_075](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_075_256_imagenet.h5)   | 2.87M  | 1.04G | 256   | 75.56    | 720.654 qps  |
  | [MobileViT_V2_100](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_100_256_imagenet.h5)   | 4.90M  | 1.83G | 256   | 78.09    | 666.113 qps  |
  | [MobileViT_V2_125](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_125_256_imagenet.h5)   | 7.48M  | 2.84G | 256   | 79.65    | 561.756 qps  |
  | [MobileViT_V2_150](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet.h5)   | 10.6M  | 4.07G | 256   | 80.38    | 489.008 qps  |
  | - [imagenet22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet22k.h5)      | 10.6M  | 4.07G | 256   | 81.46    | 489.008 qps  |
  | - [imagenet22k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_384_imagenet22k.h5) | 10.6M  | 9.15G | 384   | 82.60    | 291.174 qps  |
  | [MobileViT_V2_175](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet.h5)   | 14.3M  | 5.52G | 256   | 80.84    | 430.285 qps  |
  | - [imagenet22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet22k.h5)      | 14.3M  | 5.52G | 256   | 81.94    | 430.285 qps  |
  | - [imagenet22k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_384_imagenet22k.h5) | 14.3M  | 12.4G | 384   | 82.93    | 255.375 qps  |
  | [MobileViT_V2_200](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet.h5)   | 18.4M  | 7.12G | 256   | 81.17    | 402.205 qps  |
  | - [imagenet22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet22k.h5)      | 18.4M  | 7.12G | 256   | 82.36    | 402.205 qps  |
  | - [imagenet22k, 384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_384_imagenet22k.h5) | 18.4M  | 16.2G | 384   | 83.41    | 235.994 qps  |
## MogaNet
  - [Keras MogaNet](keras_cv_attention_models/moganet) is for [PDF 2211.03295 Efficient Multi-order Gated Aggregation Network](https://arxiv.org/pdf/2211.03295.pdf).

  | Model        | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------ | ------ | ------ | ----- | -------- | ------------ |
  | [MogaNetXtiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_xtiny_imagenet.h5) | 2.96M  | 806M   | 224   | 76.5     | 422.468 qps  |
  | [MogaNetTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_224_imagenet.h5)  | 5.20M  | 1.11G  | 224   | 79.0     | 378.634 qps  |
  | - [256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_256_imagenet.h5)        | 5.20M  | 1.45G  | 256   | 79.6     | 346.489 qps  |
  | [MogaNetSmall](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_small_imagenet.h5) | 25.3M  | 4.98G  | 224   | 83.4     | 256.359 qps  |
  | [MogaNetBase](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_base_imagenet.h5)  | 43.7M  | 9.96G  | 224   | 84.2     | 132.8 qps    |
  | [MogaNetLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_large_imagenet.h5) | 82.5M  | 15.96G | 224   | 84.6     | 87.3112 qps  |
## NAT
  - [Keras NAT](keras_cv_attention_models/nat) is for [PDF 2204.07143 Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf).

  | Model     | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | --------- | ------ | ------ | ----- | -------- | ------------ |
  | [NAT_Mini](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_mini_imagenet.h5)  | 20.0M  | 2.73G  | 224   | 81.8     |              |
  | [NAT_Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_tiny_imagenet.h5)  | 27.9M  | 4.34G  | 224   | 83.2     |              |
  | [NAT_Small](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_small_imagenet.h5) | 50.7M  | 7.84G  | 224   | 83.7     |              |
  | [NAT_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_base_imagenet.h5)  | 89.8M  | 13.76G | 224   | 84.3     |              |
## NFNets
  - [Keras NFNets](keras_cv_attention_models/nfnets) is for [PDF 2102.06171 High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171.pdf).

  | Model        | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------------ | ------ | ------- | ----- | -------- | ------------ |
  | [NFNetL0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetl0_imagenet.h5)      | 35.07M | 7.13G   | 288   | 82.75    | 293.835 qps  |
  | [NFNetF0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf0_imagenet.h5)      | 71.5M  | 12.58G  | 256   | 83.6     | 157.118 qps  |
  | [NFNetF1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf1_imagenet.h5)      | 132.6M | 35.95G  | 320   | 84.7     | 65.8114 qps  |
  | [NFNetF2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf2_imagenet.h5)      | 193.8M | 63.24G  | 352   | 85.1     | 40.0457 qps  |
  | [NFNetF3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf3_imagenet.h5)      | 254.9M | 115.75G | 416   | 85.7     | 23.9545 qps  |
  | [NFNetF4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf4_imagenet.h5)      | 316.1M | 216.78G | 512   | 85.9     | 14.9337 qps  |
  | [NFNetF5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf5_imagenet.h5)      | 377.2M | 291.73G | 544   | 86.0     | 10.4357 qps  |
  | [NFNetF6, SAM](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf6_imagenet.h5) | 438.4M | 379.75G | 576   | 86.5     | 8.4894 qps   |
  | NFNetF7      | 499.5M | 481.80G | 608   |          |              |
  | [ECA_NFNetL0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl0_imagenet.h5)  | 24.14M | 7.12G   | 288   | 82.58    | 260.699 qps  |
  | [ECA_NFNetL1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl1_imagenet.h5)  | 41.41M | 14.93G  | 320   | 84.01    | 125.579 qps  |
  | [ECA_NFNetL2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl2_imagenet.h5)  | 56.72M | 30.12G  | 384   | 84.70    | 72.0283 qps  |
  | ECA_NFNetL3  | 72.04M | 52.73G  | 448   |          | 43.501 qps   |
## PVT_V2
  - [Keras PVT_V2](keras_cv_attention_models/pvt) is for [PDF 2106.13797 PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/pdf/2106.13797.pdf).

  | Model           | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | --------------- | ------ | ------ | ----- | -------- | ------------ |
  | [PVT_V2B0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b0_imagenet.h5)        | 3.7M   | 580.3M | 224   | 70.5     | 566.962 qps  |
  | [PVT_V2B1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b1_imagenet.h5)        | 14.0M  | 2.14G  | 224   | 78.7     | 391.709 qps  |
  | [PVT_V2B2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_imagenet.h5)        | 25.4M  | 4.07G  | 224   | 82.0     | 210.827 qps  |
  | [PVT_V2B2_linear](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_linear_imagenet.h5) | 22.6M  | 3.94G  | 224   | 82.1     | 219.779 qps  |
  | [PVT_V2B3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b3_imagenet.h5)        | 45.2M  | 6.96G  | 224   | 83.1     | 137.569 qps  |
  | [PVT_V2B4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b4_imagenet.h5)        | 62.6M  | 10.19G | 224   | 83.6     | 96.5694 qps  |
  | [PVT_V2B5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b5_imagenet.h5)        | 82.0M  | 11.81G | 224   | 83.8     | 83.2803 qps  |
## RegNetY
  - [Keras RegNetY](keras_cv_attention_models/resnet_family#regnety) is for [PDF 2003.13678 Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf).

  | Model      | Params  | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ---------- | ------- | ------ | ----- | -------- | ------------ |
  | [RegNetY040](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_040_imagenet.h5) | 20.65M  | 3.98G  | 224   | 82.3     | 762.997 qps  |
  | [RegNetY064](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_064_imagenet.h5) | 30.58M  | 6.36G  | 224   | 83.0     | 426.547 qps  |
  | [RegNetY080](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_080_imagenet.h5) | 39.18M  | 7.97G  | 224   | 83.17    | 522.492 qps  |
  | [RegNetY160](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_160_imagenet.h5) | 83.59M  | 15.92G | 224   | 82.0     | 340.478 qps  |
  | [RegNetY320](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnety_320_imagenet.h5) | 145.05M | 32.29G | 224   | 82.5     | 194.246 qps  |
## RegNetZ
  - [Keras RegNetZ](keras_cv_attention_models/resnet_family#regnetz) includes implementation of [Github timm/models/byobnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py).
  - Related paper [PDF 2004.02967 Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf)

  | Model          | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | -------------- | ------ | ----- | ----- | -------- | ------------ |
  | [RegNetZB16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_b16_imagenet.h5)     | 9.72M  | 1.44G | 224   | 79.868   | 813.178 qps  |
  | [RegNetZC16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_c16_imagenet.h5)     | 13.46M | 2.50G | 256   | 82.164   | 664.265 qps  |
  | [RegNetZC16_EVO](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_c16_evo_imagenet.h5) | 13.49M | 2.55G | 256   | 81.9     |              |
  | [RegNetZD32](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d32_imagenet.h5)     | 27.58M | 5.96G | 256   | 83.422   | 466.485 qps  |
  | [RegNetZD8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d8_imagenet.h5)      | 23.37M | 3.95G | 256   | 83.5     | 477.033 qps  |
  | [RegNetZD8_EVO](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d8_evo_imagenet.h5)  | 23.46M | 4.61G | 256   | 83.42    |              |
  | [RegNetZE8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_e8_imagenet.h5)      | 57.70M | 9.88G | 256   | 84.5     | 288.301 qps  |
## ResMLP
  - [Keras ResMLP](keras_cv_attention_models/mlp_family#resmlp) includes implementation of [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf).

  | Model         | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ------- | ----- | -------- | ------------ |
  | [ResMLP12](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp12_imagenet.h5)      | 15M    | 3.02G   | 224   | 77.8     | 867.459 qps  |
  | [ResMLP24](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp24_imagenet.h5)      | 30M    | 5.98G   | 224   | 80.8     | 461.858 qps  |
  | [ResMLP36](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp36_imagenet.h5)      | 116M   | 8.94G   | 224   | 81.1     | 275.699 qps  |
  | [ResMLP_B24](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet.h5)    | 129M   | 100.39G | 224   | 83.6     | 79.4025 qps  |
  | - [imagenet22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet22k.h5) | 129M   | 100.39G | 224   | 84.4     | 79.4025 qps  |
## ResNeSt
  - [Keras ResNeSt](keras_cv_attention_models/resnest) is for [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf).

  | Model          | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------------- | ------ | ------ | ----- | -------- | ------------ |
  | [ResNest50](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50_imagenet.h5)      | 28M    | 5.38G  | 224   | 81.03    | 545.56 qps   |
  | [ResNest101](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101_imagenet.h5)     | 49M    | 13.33G | 256   | 82.83    | 275.683 qps  |
  | [ResNest200](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200_imagenet.h5)     | 71M    | 35.55G | 320   | 83.84    | 120.625 qps  |
  | [ResNest269](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269_imagenet.h5)     | 111M   | 77.42G | 416   | 84.54    | 63.6791 qps  |
## ResNetD
  - [Keras ResNetD](keras_cv_attention_models/resnet_family#resnetd) includes implementation of [PDF 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

  | Model      | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ---------- | ------ | ------ | ----- | -------- | ------------ |
  | [ResNet50D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet50d_imagenet.h5)  | 25.58M | 4.33G  | 224   | 80.530   | 906.625 qps  |
  | [ResNet101D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet101d_imagenet.h5) | 44.57M | 8.04G  | 224   | 83.022   | 485.327 qps  |
  | [ResNet152D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet152d_imagenet.h5) | 60.21M | 11.75G | 224   | 83.680   | 339.762 qps  |
  | [ResNet200D](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet200d_imagenet.h5) | 64.69M | 15.25G | 224   | 83.962   | 277.768 qps  |
## ResNetQ
  - [Keras ResNetQ](keras_cv_attention_models/resnet_family#resnetq) includes implementation of [Github timm/models/resnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py)

  | Model     | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | --------- | ------ | ----- | ----- | -------- | ------------ |
  | [ResNet51Q](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet51q_imagenet.h5) | 35.7M  | 4.87G | 224   | 82.36    | 796.151 qps  |
  | ResNet61Q | 36.8M  | 5.96G | 224   |          | 691.785 qps  |
## ResNeXt
  - [Keras ResNeXt](keras_cv_attention_models/resnet_family#resnext) includes implementation of [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
  - `SWSL` means `Semi-Weakly Supervised ResNe*t` from [Github facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models). **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.

  | Model                      | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [ResNeXt50, (32x4d)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_imagenet.h5)         | 25M    | 4.23G  | 224   | 79.768   | 1017.99 qps  |
  | - [SWSL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_swsl.h5)                     | 25M    | 4.23G  | 224   | 82.182   | 1017.99 qps  |
  | [ResNeXt50D, (32x4d + deep)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50d_imagenet.h5) | 25M    | 4.47G  | 224   | 79.676   | 975.949 qps  |
  | [ResNeXt101, (32x4d)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)        | 42M    | 7.97G  | 224   | 80.334   | 551.088 qps  |
  | - [SWSL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_swsl.h5)                     | 42M    | 7.97G  | 224   | 83.230   | 551.088 qps  |
  | [ResNeXt101W, (32x8d)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)       | 89M    | 16.41G | 224   | 79.308   | 357.177 qps  |
  | - [SWSL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101w_swsl.h5)                     | 89M    | 16.41G | 224   | 84.284   | 357.177 qps  |
  | [ResNeXt101W_64, (64x4d)](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101w_64_imagenet.h5)    | 83.46M | 15.46G | 224   | 82.46    | 363.869 qps  |
## SwinTransformerV2
  - [Keras SwinTransformerV2](keras_cv_attention_models/swin_transformer_v2) includes implementation of [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).

  | Model                                | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------------------------------ | ------ | ------ | ----- | -------- | ------------ |
  | [SwinTransformerV2Tiny_ns](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_ns_224_imagenet.h5)             | 28.3M  | 4.69G  | 224   | 81.8     | 292.892 qps  |
  | [SwinTransformerV2Small_ns](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_ns_224_imagenet.h5)            | 49.7M  | 9.12G  | 224   | 83.5     | 164.114 qps  |
  | [SwinTransformerV2Tiny_window8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window8_256_imagenet.h5)        | 28.3M  | 5.99G  | 256   | 81.8     | 266.172 qps  |
  | [SwinTransformerV2Tiny_window16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window16_256_imagenet.h5)       | 28.3M  | 6.75G  | 256   | 82.8     | 207.568 qps  |
  | [SwinTransformerV2Small_window8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window8_256_imagenet.h5)       | 49.7M  | 11.63G | 256   | 83.7     | 141.093 qps  |
  | [SwinTransformerV2Small_window16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window16_256_imagenet.h5)      | 49.7M  | 12.93G | 256   | 84.1     | 122.444 qps  |
  | [SwinTransformerV2Base_window8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window8_256_imagenet.h5)        | 87.9M  | 20.44G | 256   | 84.2     | 118.361 qps  |
  | [SwinTransformerV2Base_window16](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet.h5)       | 87.9M  | 22.17G | 256   | 84.6     | 94.8596 qps  |
  | [SwinTransformerV2Base_window16, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet22k.h5)  | 87.9M  | 22.17G | 256   | 86.2     | 94.8596 qps  |
  | [SwinTransformerV2Base_window24, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window24_384_imagenet22k.h5)  | 87.9M  | 55.89G | 384   | 87.1     | 35.4368 qps  |
  | [SwinTransformerV2Large_window16, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window16_256_imagenet22k.h5) | 196.7M | 48.03G | 256   | 86.9     | 62.1343 qps  |
  | [SwinTransformerV2Large_window24, 22k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window24_384_imagenet22k.h5) | 196.7M | 117.1G | 384   | 87.6     | 21.8255 qps  |
## TinyNet
  - [Keras TinyNet](keras_cv_attention_models/mobilenetv3_family#tinynet) includes implementation of [PDF 2010.14819 Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets](https://arxiv.org/pdf/2010.14819.pdf).

  | Model    | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | -------- | ------ | ------- | ----- | -------- | ------------ |
  | [TinyNetE](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_e_imagenet.h5) | 2.04M  | 25.22M  | 106   | 59.86    | 2460.12 qps  |
  | [TinyNetD](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_d_imagenet.h5) | 2.34M  | 53.35M  | 152   | 66.96    | 2165.13 qps  |
  | [TinyNetC](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_c_imagenet.h5) | 2.46M  | 103.22M | 184   | 71.23    | 1550.45 qps  |
  | [TinyNetB](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_b_imagenet.h5) | 3.73M  | 206.28M | 188   | 74.98    | 1326.3 qps   |
  | [TinyNetA](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_a_imagenet.h5) | 6.19M  | 343.74M | 192   | 77.65    | 1076.39 qps  |
## TinyViT
  - [Keras TinyViT](keras_cv_attention_models/tinyvit) includes implementation of [PDF 2207.10666 TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf).

  | Model                | Params | FLOPs | Input | Top1 Acc | T4 Inference |
  | -------------------- | ------ | ----- | ----- | -------- | ------------ |
  | [TinyViT_5M, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet.h5)  | 5.4M   | 1.3G  | 224   | 79.1     | 587.78 qps   |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet21k-ft1k.h5)   | 5.4M   | 1.3G  | 224   | 80.7     | 587.78 qps   |
  | [TinyViT_11M, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet.h5) | 11M    | 2.0G  | 224   | 81.5     | 454.312 qps  |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet21k-ft1k.h5)   | 11M    | 2.0G  | 224   | 83.2     | 454.312 qps  |
  | [TinyViT_21M, distill](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet.h5) | 21M    | 4.3G  | 224   | 83.1     | 334.34 qps   |
  | - [imagenet21k-ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet21k-ft1k.h5)   | 21M    | 4.3G  | 224   | 84.8     | 334.34 qps   |
  | - [384, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_384_imagenet21k-ft1k.h5)           | 21M    | 13.8G | 384   | 86.2     | 201.022 qps  |
  | - [512, 21k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_512_imagenet21k-ft1k.h5)           | 21M    | 27.0G | 512   | 86.5     | 126.521 qps  |
## UniFormer
  - [Keras UniFormer](keras_cv_attention_models/uniformer) includes implementation of [PDF 2201.09450 UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/pdf/2201.09450.pdf).

  | Model                | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | -------------------- | ------ | ------ | ----- | -------- | ------------ |
  | [UniformerSmall32, TL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_32_224_token_label.h5) | 22M    | 3.66G  | 224   | 83.4     | 414.23 qps   |
  | [UniformerSmall64](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_imagenet.h5)     | 22M    | 3.66G  | 224   | 82.9     | 408.464 qps  |
  | - [Token Labeling](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_token_label.h5)     | 22M    | 3.66G  | 224   | 83.4     | 408.464 qps  |
  | [UniformerSmallPlus32](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_imagenet.h5) | 24M    | 4.24G  | 224   | 83.4     | 377.856 qps  |
  | - [Token Labeling](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_token_label.h5)     | 24M    | 4.24G  | 224   | 83.9     | 377.856 qps  |
  | [UniformerSmallPlus64](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_imagenet.h5) | 24M    | 4.23G  | 224   | 83.4     | 373.724 qps  |
  | - [Token Labeling](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_token_label.h5)     | 24M    | 4.23G  | 224   | 83.6     | 373.724 qps  |
  | [UniformerBase32, TL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_32_224_token_label.h5)  | 50M    | 8.32G  | 224   | 85.1     | 186.922 qps  |
  | [UniformerBase64](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_imagenet.h5)      | 50M    | 8.31G  | 224   | 83.8     | 187.371 qps  |
  | - [Token Labeling](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_token_label.h5)     | 50M    | 8.31G  | 224   | 84.8     | 187.371 qps  |
  | [UniformerLarge64, TL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_224_token_label.h5) | 100M   | 19.79G | 224   | 85.6     | 105.681 qps  |
  | - [384, TL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_384_token_label.h5)            | 100M   | 63.11G | 384   | 86.3     | 52.4368 qps  |
## VanillaNet
  - [Keras VanillaNet](keras_cv_attention_models/vanillanet) is for [PDF 2305.12972 VanillaNet: the Power of Minimalism in Deep Learning](https://arxiv.org/pdf/2305.12972.pdf).

  | Model         | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | ------------- | ------ | ------ | ----- | -------- | ------------ |
  | [VanillaNet5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_5_imagenet.h5)   | 22.33M | 8.46G  | 224   | 72.49    | 605.05 qps   |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_5_deploy_imagenet.h5) | 15.52M | 5.17G  | 224   | 72.49    | 784.606 qps  |
  | [VanillaNet6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_6_imagenet.h5)   | 56.12M | 10.11G | 224   | 76.36    | 507.217 qps  |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_6_deploy_imagenet.h5) | 32.51M | 6.00G  | 224   | 76.36    | 676.752 qps  |
  | [VanillaNet7](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_7_imagenet.h5)   | 56.67M | 11.84G | 224   | 77.98    | 408.414 qps  |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_7_deploy_imagenet.h5) | 32.80M | 6.90G  | 224   | 77.98    | 550.639 qps  |
  | [VanillaNet8](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_8_imagenet.h5)   | 65.18M | 13.50G | 224   | 79.13    | 370.308 qps  |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_8_deploy_imagenet.h5) | 37.10M | 7.75G  | 224   | 79.13    | 483.333 qps  |
  | [VanillaNet9](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_9_imagenet.h5)   | 73.68M | 15.17G | 224   | 79.87    | 336.336 qps  |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_9_deploy_imagenet.h5) | 41.40M | 8.59G  | 224   | 79.87    | 473.191 qps  |
  | [VanillaNet10](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_10_imagenet.h5)  | 82.19M | 16.83G | 224   | 80.57    | 303.437 qps  |
  | - [deploy=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_10_deploy_imagenet.h5) | 45.69M | 9.43G  | 224   | 80.57    | 425.838 qps  |
  | VanillaNet11  | 90.69M | 18.49G | 224   | 81.08    | 276.779 qps  |
  | - deploy=True | 50.00M | 10.27G | 224   | 81.08    | 391.48 qps   |
  | VanillaNet12  | 99.20M | 20.16G | 224   | 81.55    | 257.595 qps  |
  | - deploy=True | 54.29M | 11.11G | 224   | 81.55    | 359.306 qps  |
  | VanillaNet13  | 107.7M | 21.82G | 224   | 82.05    | 240.025 qps  |
  | - deploy=True | 58.59M | 11.96G | 224   | 82.05    | 338.175 qps  |
## VOLO
  - [Keras VOLO](keras_cv_attention_models/volo) is for [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).

  | Model   | Params | FLOPs   | Input | Top1 Acc | T4 Inference |
  | ------- | ------ | ------- | ----- | -------- | ------------ |
  | [VOLO_d1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_224_imagenet.h5) | 27M    | 4.82G   | 224   | 84.2     |              |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_384_imagenet.h5)   | 27M    | 14.22G  | 384   | 85.2     |              |
  | [VOLO_d2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_224_imagenet.h5) | 59M    | 9.78G   | 224   | 85.2     |              |
  | - [384](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_384_imagenet.h5)   | 59M    | 28.84G  | 384   | 86.0     |              |
  | [VOLO_d3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_224_imagenet.h5) | 86M    | 13.80G  | 224   | 85.4     |              |
  | - [448](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_448_imagenet.h5)   | 86M    | 55.50G  | 448   | 86.3     |              |
  | [VOLO_d4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_224_imagenet.h5) | 193M   | 29.39G  | 224   | 85.7     |              |
  | - [448](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_448_imagenet.h5)   | 193M   | 117.81G | 448   | 86.8     |              |
  | [VOLO_d5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_224_imagenet.h5) | 296M   | 53.34G  | 224   | 86.1     |              |
  | - [448](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_448_imagenet.h5)   | 296M   | 213.72G | 448   | 87.0     |              |
  | - [512](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_512_imagenet.h5)   | 296M   | 279.36G | 512   | 87.1     |              |
## WaveMLP
  - [Keras WaveMLP](keras_cv_attention_models/mlp_family#wavemlp) includes implementation of [PDF 2111.12294 An Image Patch is a Wave: Quantum Inspired Vision MLP](https://arxiv.org/pdf/2111.12294.pdf).

  | Model     | Params | FLOPs  | Input | Top1 Acc | T4 Inference |
  | --------- | ------ | ------ | ----- | -------- | ------------ |
  | [WaveMLP_T](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_t_imagenet.h5) | 17M    | 2.47G  | 224   | 80.9     | 557.307 qps  |
  | [WaveMLP_S](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_s_imagenet.h5) | 30M    | 4.55G  | 224   | 82.9     | 249.051 qps  |
  | [WaveMLP_M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/wavemlp_m_imagenet.h5) | 44M    | 7.92G  | 224   | 83.3     | 166.673 qps  |
  | WaveMLP_B | 63M    | 10.26G | 224   | 83.6     | 160.065 qps  |
***

# Detection Models
## EfficientDet
  - [Keras EfficientDet](keras_cv_attention_models/efficientdet) includes implementation of [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).
  - `Det-AdvProp + AutoAugment` [Paper 2103.13886 Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/pdf/2103.13886.pdf).

  | Model              | Params | FLOPs   | Input | COCO val AP | test AP | T4 Inference |
  | ------------------ | ------ | ------- | ----- | ----------- | ------- | ------------ |
  | [EfficientDetD0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d0_512_coco.h5)     | 3.9M   | 2.55G   | 512   | 34.3        | 34.6    |              |
  | - Det-AdvProp      | 3.9M   | 2.55G   | 512   | 35.1        | 35.3    |              |
  | [EfficientDetD1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d1_640_coco.h5)     | 6.6M   | 6.13G   | 640   | 40.2        | 40.5    |              |
  | - Det-AdvProp      | 6.6M   | 6.13G   | 640   | 40.8        | 40.9    |              |
  | [EfficientDetD2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d2_768_coco.h5)     | 8.1M   | 11.03G  | 768   | 43.5        | 43.9    |              |
  | - Det-AdvProp      | 8.1M   | 11.03G  | 768   | 44.3        | 44.3    |              |
  | [EfficientDetD3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d3_896_coco.h5)     | 12.0M  | 24.95G  | 896   | 46.8        | 47.2    |              |
  | - Det-AdvProp      | 12.0M  | 24.95G  | 896   | 47.7        | 48.0    |              |
  | [EfficientDetD4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d4_1024_coco.h5)     | 20.7M  | 55.29G  | 1024  | 49.3        | 49.7    |              |
  | - Det-AdvProp      | 20.7M  | 55.29G  | 1024  | 50.4        | 50.4    |              |
  | [EfficientDetD5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d5_1280_coco.h5)     | 33.7M  | 135.62G | 1280  | 51.2        | 51.5    |              |
  | - Det-AdvProp      | 33.7M  | 135.62G | 1280  | 52.2        | 52.5    |              |
  | [EfficientDetD6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d6_1280_coco.h5)     | 51.9M  | 225.93G | 1280  | 52.1        | 52.6    | 11.3327 qps  |
  | [EfficientDetD7](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7_1536_coco.h5)     | 51.9M  | 325.34G | 1536  | 53.4        | 53.7    | 7.81326 qps  |
  | [EfficientDetD7X](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7x_1536_coco.h5)    | 77.0M  | 410.87G | 1536  | 54.4        | 55.1    | 6.37054 qps  |
  | [EfficientDetLite0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite0_320_coco.h5)  | 3.2M   | 0.98G   | 320   | 27.5        | 26.41   | 664.793 qps  |
  | [EfficientDetLite1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite1_384_coco.h5)  | 4.2M   | 1.97G   | 384   | 32.6        | 31.50   | 445.699 qps  |
  | [EfficientDetLite2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite2_448_coco.h5)  | 5.3M   | 3.38G   | 448   | 36.2        | 35.06   | 323.178 qps  |
  | [EfficientDetLite3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3_512_coco.h5)  | 8.4M   | 7.50G   | 512   | 39.9        | 38.77   | 197.332 qps  |
  | [EfficientDetLite3X](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3x_640_coco.h5) | 9.3M   | 14.01G  | 640   | 44.0        | 42.64   | 123.29 qps   |
  | [EfficientDetLite4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite4_640_coco.h5)  | 15.1M  | 20.20G  | 640   | 44.4        | 43.18   | 101.555 qps  |
## YOLO_NAS
  - [Keras YOLO_NAS](keras_cv_attention_models/yolov8) includes implementation of [Github Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) YOLO-NAS models.

  | Model                   | Params | FLOPs  | Input | COCO val AP | test AP | T4 Inference |
  | ----------------------- | ------ | ------ | ----- | ----------- | ------- | ------------ |
  | [YOLO_NAS_S](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_s_coco.h5)              | 12.18M | 15.92G | 640   | 47.5        |         | 320.124 qps  |
  | - [use_reparam_conv=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_s_before_reparam_coco.h5) | 12.88M | 16.96G | 640   | 47.5        |         | 239.282 qps  |
  | [YOLO_NAS_M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_m_coco.h5)              | 31.92M | 43.91G | 640   | 51.55       |         | 166.667 qps  |
  | - [use_reparam_conv=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_m_before_reparam_coco.h5) | 33.86M | 47.12G | 640   | 51.55       |         | 128.672 qps  |
  | [YOLO_NAS_L](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_l_coco.h5)              | 42.02M | 59.95G | 640   | 52.22       |         | 129.892 qps  |
  | - [use_reparam_conv=True](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_l_before_reparam_coco.h5) | 44.53M | 64.53G | 640   | 52.22       |         | 97.828 qps   |
## YOLOR
  - [Keras YOLOR](keras_cv_attention_models/yolor) includes implementation of [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).

  | Model      | Params | FLOPs   | Input | COCO val AP | test AP | T4 Inference |
  | ---------- | ------ | ------- | ----- | ----------- | ------- | ------------ |
  | [YOLOR_CSP](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_coco.h5)  | 52.9M  | 60.25G  | 640   | 50.0        | 52.8    | 120.886 qps  |
  | [YOLOR_CSPX](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_x_coco.h5) | 99.8M  | 111.11G | 640   | 51.5        | 54.8    | 64.5295 qps  |
  | [YOLOR_P6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_p6_coco.h5)   | 37.3M  | 162.87G | 1280  | 52.5        | 55.7    | 52.0692 qps  |
  | [YOLOR_W6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_w6_coco.h5)   | 79.9M  | 226.67G | 1280  | 53.6 ?      | 56.9    | 41.8772 qps  |
  | [YOLOR_E6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_e6_coco.h5)   | 115.9M | 341.62G | 1280  | 50.3 ?      | 57.6    | 23.2091 qps  |
  | [YOLOR_D6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_d6_coco.h5)   | 151.8M | 467.88G | 1280  | 50.8 ?      | 58.2    | 17.4138 qps  |
## YOLOV7
  - [Keras YOLOV7](keras_cv_attention_models/yolov7) includes implementation of [Paper 2207.02696 YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf).

  | Model       | Params | FLOPs  | Input | COCO val AP | test AP | T4 Inference |
  | ----------- | ------ | ------ | ----- | ----------- | ------- | ------------ |
  | [YOLOV7_Tiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_tiny_coco.h5) | 6.23M  | 2.90G  | 416   | 33.3        |         | 857.615 qps  |
  | [YOLOV7_CSP](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_csp_coco.h5)  | 37.67M | 53.0G  | 640   | 51.4        |         | 139.919 qps  |
  | [YOLOV7_X](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_x_coco.h5)    | 71.41M | 95.0G  | 640   | 53.1        |         | 83.818 qps   |
  | [YOLOV7_W6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_w6_coco.h5)   | 70.49M | 180.1G | 1280  | 54.9        |         | 52.5463 qps  |
  | [YOLOV7_E6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6_coco.h5)   | 97.33M | 257.6G | 1280  | 56.0        |         | 32.5041 qps  |
  | [YOLOV7_D6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_d6_coco.h5)   | 133.9M | 351.4G | 1280  | 56.6        |         | 27.9053 qps  |
  | [YOLOV7_E6E](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6e_coco.h5)  | 151.9M | 421.7G | 1280  | 56.8        |         | 20.9202 qps  |
## YOLOV8
  - [Keras YOLOV8](keras_cv_attention_models/yolov8) includes implementation of [Github ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) detection and classification models.

  | Model     | Params | FLOPs  | Input | COCO val AP | test AP | T4 Inference |
  | --------- | ------ | ------ | ----- | ----------- | ------- | ------------ |
  | [YOLOV8_N](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_n_coco.h5)  | 3.16M  | 4.39G  | 640   | 37.3        |         | 622.29 qps   |
  | [YOLOV8_S](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_s_coco.h5)  | 11.17M | 14.33G | 640   | 44.9        |         | 361.696 qps  |
  | [YOLOV8_M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_m_coco.h5)  | 25.90M | 39.52G | 640   | 50.2        |         | 160.866 qps  |
  | [YOLOV8_L](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_l_coco.h5)  | 43.69M | 82.65G | 640   | 52.9        |         | 105.888 qps  |
  | [YOLOV8_X](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_x_coco.h5)  | 68.23M | 129.0G | 640   | 53.9        |         | 66.6191 qps  |
  | [YOLOV8_X6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_x6_coco.h5) | 97.42M | 522.6G | 1280  | 56.7 ?      |         | 17.9007 qps  |
## YOLOX
  - [Keras YOLOX](keras_cv_attention_models/yolox) includes implementation of [Paper 2107.08430 YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf).

  | Model     | Params | FLOPs   | Input | COCO val AP | test AP | T4 Inference |
  | --------- | ------ | ------- | ----- | ----------- | ------- | ------------ |
  | [YOLOXNano](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_nano_coco.h5) | 0.91M  | 0.53G   | 416   | 25.8        |         | 975.445 qps  |
  | [YOLOXTiny](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_tiny_coco.h5) | 5.06M  | 3.22G   | 416   | 32.8        |         | 734.848 qps  |
  | [YOLOXS](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_s_coco.h5)    | 9.0M   | 13.39G  | 640   | 40.5        | 40.5    | 376.623 qps  |
  | [YOLOXM](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_m_coco.h5)    | 25.3M  | 36.84G  | 640   | 46.9        | 47.2    | 172.361 qps  |
  | [YOLOXL](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_l_coco.h5)    | 54.2M  | 77.76G  | 640   | 49.7        | 50.1    | 111.519 qps  |
  | [YOLOXX](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_x_coco.h5)    | 99.1M  | 140.87G | 640   | 51.5        | 51.5    | 62.6076 qps  |
***

# Language Models
## GPT2
  - [Keras GPT2](keras_cv_attention_models/gpt2) includes implementation of [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

  | Model            | Params  | FLOPs   | vocab_size | LAMBADA PPL | T4 Inference |
  | ---------------- | ------- | ------- | ---------- | ----------- | ------------ |
  | [GPT2_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_base_webtext.h5)        | 163.04M | 146.42G | 50257      | 35.13       |              |
  | [GPT2_Medium](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_medium_webtext.h5)      | 406.29M | 415.07G | 50257      | 15.60       |              |
  | [GPT2_Large](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_large_webtext.h5)       | 838.36M | 890.28G | 50257      | 10.87       |              |
  | [GPT2_XLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_xlarge_webtext.1.h5), [+.2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_xlarge_webtext.2.h5) | 1.638B  | 1758.3G | 50257      | 8.63        |              |
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
