# ___Keras YOLOV8___
- [Summary](#summary)
- [Detection Models](#detection-models)
- [Classification Models](#classification-models)
- [Usage](#usage)
- [Custom detector using YOLOV8 header](#custom-detector-using-yolov8-header)
- [Verification with PyTorch version](#verification-with-pytorch-version)
- [COCO eval results](#coco-eval-results)
- [Training using PyTorch backend and ultralytics](#training-using-pytorch-backend-and-ultralytics)
***

## Summary
  - Keras implementation of [Github ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) detection and classification models. Model weights converted from official publication.
***

## Detection Models
  | Model     | Params | FLOPs  | Input | COCO val AP | test AP | Download |
  | --------- | ------ | ------ | ----- | ----------- | ------- | -------- |
  | YOLOV8_N  | 3.16M  | 4.39G  | 640   | 37.3        |         | [yolov8_n_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_n_coco.h5) |
  | YOLOV8_S  | 11.17M | 14.33G | 640   | 44.9        |         | [yolov8_s_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_s_coco.h5) |
  | YOLOV8_M  | 25.90M | 39.52G | 640   | 50.2        |         | [yolov8_m_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_m_coco.h5) |
  | YOLOV8_L  | 43.69M | 82.65G | 640   | 52.9        |         | [yolov8_l_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_l_coco.h5) |
  | YOLOV8_X  | 68.23M | 129.0G | 640   | 53.9        |         | [yolov8_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_x_coco.h5) |
  | YOLOV8_X6 | 97.42M | 522.6G | 1280  | 56.7 ?      |         | [yolov8_x6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_x6_coco.h5) |

  | Model                    | Params | FLOPs  | Input | COCO val AP | test AP | Download |
  | ------------------------ | ------ | ------ | ----- | ----------- | ------- | -------- |
  | YOLO_NAS_S               | 12.88M | 16.96G | 640   | 47.5        |         | [s_before_reparam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_s_before_reparam_coco.h5) |
  | - use_reparam_conv=False | 12.18M | 15.92G | 640   | 47.5        |         | [yolo_nas_s_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_s_coco.h5) |
  | YOLO_NAS_M               | 33.86M | 47.12G | 640   | 51.55       |         | [m_before_reparam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_m_before_reparam_coco.h5) |
  | - use_reparam_conv=False | 31.92M | 43.91G | 640   | 51.55       |         | [yolo_nas_m_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_m_coco.h5) |
  | YOLO_NAS_L               | 44.53M | 64.53G | 640   | 52.22       |         | [l_before_reparam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_l_before_reparam_coco.h5) |
  | - use_reparam_conv=False | 42.02M | 59.95G | 640   | 52.22       |         | [yolo_nas_l_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolo_nas_l_coco.h5) |
## Classification Models
  | Model        | Params | FLOPs@640 | FLOPs@224 | Input | Top1 Acc | Download |
  | ------------ | ------ | --------- | --------- | ----- | -------- | -------- |
  | YOLOV8_N_CLS | 2.72M  | 1.65G     | 203.7M    | 224   | 66.6     | [yolov8_n_cls.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_n_cls_imagenet.h5) |
  | YOLOV8_S_CLS | 6.36M  | 6.24G     | 765.7M    | 224   | 72.3     | [yolov8_s_cls.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_s_cls_imagenet.h5) |
  | YOLOV8_M_CLS | 17.05M | 20.85G    | 2.56G     | 224   | 76.4     | [yolov8_m_cls.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_m_cls_imagenet.h5) |
  | YOLOV8_L_CLS | 37.48M | 49.41G    | 6.05G     | 224   | 78.0     | [yolov8_l_cls.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_l_cls_imagenet.h5) |
  | YOLOV8_X_CLS | 57.42M | 76.96G    | 9.43G     | 224   | 78.4     | [yolov8_x_cls.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov8/yolov8_x_cls_imagenet.h5) |
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import yolov8
    model = yolov8.YOLOV8_N(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences)
    ```
    ![yolov8_n_dog_cat](https://user-images.githubusercontent.com/5744524/230085258-14aee245-0084-4090-a62f-a2f23ce800f5.png)
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`. **Note: For `YOLO_NAS` models, actual input shape needs to be divisible by `32`**.
    ```py
    from keras_cv_attention_models import yolov8
    model = yolov8.YOLOV8_S(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: ~/.keras/models/yolov8_s_coco.h5
    print(model.input_shape, model.output_shape)
    # (None, None, None, 3) (None, None, 144)
    print(model(tf.ones([1, 742, 355, 3])).shape)
    # (1, 5554, 144)
    print(model(tf.ones([1, 188, 276, 3])).shape)
    # (1, 1110, 144)

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (320, 224, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
    ```
    ![yolov8_s_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/230587610-8a276623-2ec9-49f1-a678-998b913a0739.png)
  - **Switch to deploy** by calling `model.switch_to_deploy()` if using `use_reparam_conv=True`. Will fuse reparameter block into a single `Conv2D` layer. Also applying `convert_to_fused_conv_bn_model` that fusing `Conv2D->BatchNorm`.
    ```py
    from keras_cv_attention_models import yolov8, test_images, model_surgery

    mm = yolov8.YOLO_NAS_S(use_reparam_conv=True)
    model_surgery.count_params(mm)
    # Total params: 12,911,584.0 | Trainable params: 12,878,304.0 | Non-trainable params:33,280.0
    preds = mm(mm.preprocess_input(test_images.dog_cat()))

    bb = mm.switch_to_deploy()
    model_surgery.count_params(bb)
    # Total params: 12,167,600.0 | Trainable params: 12,167,600.0 | Non-trainable params:0.0
    preds_deploy = bb(bb.preprocess_input(test_images.dog_cat()))

    print(f"{np.allclose(preds, preds_deploy, atol=1e-3) = }")
    # np.allclose(preds, preds_deploy, atol=1e-3) = True
    ```
  - **Classification model**
    ```py
    from keras_cv_attention_models.yolov8 import yolov8
    model = yolov8.YOLOV8_N_CLS(pretrained="imagenet")

    # Run prediction
    from skimage.data import chelsea # Chelsea the cat
    preds = model(model.preprocess_input(chelsea()))
    print(model.decode_predictions(preds))
    # [('n02124075', 'Egyptian_cat', 0.2490207), ('n02123045', 'tabby', 0.12989485), ...]
    ```
  - **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
    ```py
    os.environ['KECAM_BACKEND'] = 'torch'

    from keras_cv_attention_models import yolov8
    model = yolov8.YOLOV8_S(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Using PyTorch backend
    # >>>> Aligned input_shape: [3, None, None]
    # >>>> Load pretrained from: ~/.keras/models/yolov8_s_coco.h5

    print(model.input_shape, model.output_shape)
    # (None, 3, None, None) (None, None, 144)

    import torch
    print(model(torch.ones([1, 3, 736, 352])).shape)
    # torch.Size([1, 5313, 144])

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (320, 224, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
    ```
## Custom detector using YOLOV8 header
  - **Backbone** for `YOLOV8` can be any model with pyramid stage structure. **NOTE: `YOLOV8` has a default `regression_len=64` for bbox output length. Typically it's `4` for other detection models, for yolov8 it's `reg_max=16 -> regression_len = 16 * 4 == 64`.**
    ```py
    from keras_cv_attention_models import efficientnet, yolov8
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)
    mm = yolov8.YOLOV8(backbone=bb)
    # >>>> features: {'stack_2_block2_output': (None, 32, 32, 48),
    #                 'stack_4_block5_output': (None, 16, 16, 112),
    #                 'stack_5_block8_output': (None, 8, 8, 192)}

    mm.summary()  # Trainable params: 8,025,252
    print(mm.output_shape)
    # (None, 1344, 144)
    ```
  - Currently 4 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor", "yolov8"]`. Default is `"yolov8"`.
    ```py
    from keras_cv_attention_models import efficientnet, yolov8
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)

    mm = yolov8.YOLOV8(backbone=bb, anchors_mode="anchor_free", regression_len=4) # Trainable params: 7,756,707
    print(mm.output_shape) # (None, 1344, 85)

    mm = yolov8.YOLOV8(backbone=bb, anchors_mode="efficientdet", regression_len=64) # Trainable params: 8,280,612
    print(mm.output_shape) # (None, 1344, 1296) -> 1296 == num_anchors 9 * (regression_len 64 + num_classes 80)
    ```
    **Default settings for anchors_mode**

    | anchors_mode | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------ | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free  | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor        | True              | 3           | None         | presets       | None       | offset=0.5      |
    | yolov8       | False             | 1           | 1            | [1]           | 1          | False           |
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 640, 640, 3)).astype("float32")

  """ PyTorch yolov8n """
  sys.path.append('../ultralytics')
  import torch

  tt = torch.load('yolov8n.pt')
  _ = tt['model'].eval()
  torch_model = tt['model'].float()
  _, torch_out = torch_model(torch.from_numpy(inputs).permute([0, 3, 1, 2]))
  torch_out_concat = [ii.reshape([1, ii.shape[1], -1]) for ii in torch_out]
  torch_out_concat = torch.concat(torch_out_concat, axis=-1).permute([0, 2, 1])

  """ Keras YOLOV8_N """
  from keras_cv_attention_models import yolov8
  mm = yolov8.YOLOV8_N(pretrained='coco', classifier_activation=None)
  keras_out = mm(inputs)

  """ Model outputs verification """
  # [top, left, bottom, right] -> [left, top, right, bottom]
  bbox_out, cls_out = tf.split(keras_out, [64, 80], axis=-1)
  bbox_out = tf.gather(tf.reshape(bbox_out, [1, -1, 4, 16]), [1, 0, 3, 2], axis=-2)
  bbox_out = tf.reshape(bbox_out, [1, -1, 4 * 16])
  keras_out_reorder = tf.concat([bbox_out, cls_out], axis=-1)
  print(f"{np.allclose(keras_out_reorder, torch_out_concat.detach(), atol=1e-4) = }")
  # np.allclose(keras_out_reorder, torch_out_concat.detach(), atol=1e-4) = True
  ```
## COCO eval results
  ```sh
  python coco_eval_script.py -m yolov8.YOLOV8_N --nms_method hard --nms_iou_or_sigma 0.65 --nms_max_output_size 300 \
  --nms_topk -1 --letterbox_pad 64 --input_shape 704
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
  # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.529
  # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.585
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.355
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.649
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.761
  ```
  ```sh
  python coco_eval_script.py -m yolov8.YOLOV8_X6 --nms_method hard --nms_iou_or_sigma 0.65 --nms_max_output_size 300 \
  --nms_topk -1 --letterbox_pad 64 --input_shape 1344
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.567
  # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.740
  # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.618
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.428
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.612
  # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.410
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.688
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.739
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.623
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.772
  # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.855
  ```
## Training using PyTorch backend and ultralytics
  - **[Experimental] Training using PyTorch backend**, currently using `ultralytics` dataset and validator process. The advantage is that this supports any pyramid staged model in this package.
  - The parameter `rect_val=False` means using fixed data shape `[640, 640]` for validator, or will by dynamic.
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
***
