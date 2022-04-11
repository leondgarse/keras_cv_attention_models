# ___COCO___
***
## Data
  - Default data augment in `coco_train_script.py` is `mosaic mix prob=0.5` + `randaug magnitude=6 with rotate, shear, transpose`.
    - `mosaic_mix_prob` is applied per batch, means each image is repeated `4` times, then randomly shuffled and mosaic mixup with others in an entire batch.
    - `color_augment_method` is one of `["random_hsv", "autoaug", "randaug"]`. For `autoaug` and `randaug` using only none-positional related methods.
    - `positional_augment_methods` is a combination of `r: rotate, t: translate, s: shear, x: scale_x + scale_y`. `None` or `''` for aspect aware random scale only.
    ```py
    from keras_cv_attention_models.coco import data
    """ random_hsv + random scale """
    # set use_anchor_free_mode=True will just return original bboxes
    tt = data.init_dataset(magnitude=10, positional_augment_methods=None, use_anchor_free_mode=True, batch_size=4)[0]
    fig = data.show_batch_sample(tt, use_anchor_free_mode=True, rows=1)

    """ random_hsv + random rotate / translate / shear / scale """
    tt = data.init_dataset(magnitude=6, positional_augment_methods='rts', use_anchor_free_mode=True, batch_size=4)[0]
    fig = data.show_batch_sample(tt, use_anchor_free_mode=True, rows=1)

    """ autoaug + random translate / scale_x / scale_y """
    tt = data.init_dataset(magnitude=6, color_augment_method='autoaug', positional_augment_methods='tx', use_anchor_free_mode=True, batch_size=4)[0]
    fig = data.show_batch_sample(tt, use_anchor_free_mode=True, rows=1)

    """ Mosaic mix + randaug + random rotate / shear / scale """
    tt = data.init_dataset(magnitude=6, mosaic_mix_prob=1.0, color_augment_method='randaug', positional_augment_methods='rs', use_anchor_free_mode=True, batch_size=4)[0]
    fig = data.show_batch_sample(tt, use_anchor_free_mode=True, rows=1)
    ```
    ![coco_data_aug_3](https://user-images.githubusercontent.com/5744524/162143972-d2d752e6-5702-42d7-9ff0-1243d2c28566.png)
  - **TFDS COCO data format**, `bboxes` in format `[top, left, bottom, right]` with value range in `[0, 1]`. It's the default compatible data format for this package.
    ```py
    import tensorflow_datasets as tfds
    ds, info = tfds.load('coco/2017', with_info=True)
    aa = ds['train'].as_numpy_iterator().next()
    print(aa['image'].shape)
    # (462, 640, 3)
    print(aa['objects'])
    # {'area': array([17821, 16942,  4344]),
    #  'bbox': array([[0.54380953, 0.13464062, 0.98651516, 0.33742186],
    #         [0.50707793, 0.517875  , 0.8044805 , 0.891125  ],
    #         [0.3264935 , 0.36971876, 0.65203464, 0.4431875 ]], dtype=float32),
    #  'id': array([152282, 155195, 185150]),
    #  'is_crowd': array([False, False, False]),
    #  'label': array([3, 3, 0])}

    imm = aa['image']
    plt.imshow(imm)

    for bb in aa["objects"]["bbox"]:
        bb = np.array([bb[0] * imm.shape[0], bb[1] * imm.shape[1], bb[2] * imm.shape[0], bb[3] * imm.shape[1]])
        plt.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])
    ```
## Training
  - `AnchorFreeLoss` usage took me weeks solving why the `bbox_loss` always been `1`. that using `tf.stop_gradient` while assigning is the key...
  - Default parameters for `coco_train_script.py` is `EfficientDetD0` with `input_shape=(256, 256, 3), batch_size=64, mosaic_mix_prob=0.5, freeze_backbone_epochs=32, total_epochs=105`. Technically, it's any `pyramid structure backbone` + `EfficientDet / YOLOX header / YOLOR header` + `anchor_free / yolor_anchors / efficientdet_anchors` combination supported.
  - Currently 3 types anchors supported,
    - **use_anchor_free_mode** controls if using typical `YOLOX anchor_free mode` strategy.
    - **use_yolor_anchors_mode** controls if using yolor anchors.
    - Default is `use_anchor_free_mode=False, use_yolor_anchors_mode=False`, means using `efficientdet` preset anchors.

    | anchors_mode  | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------- | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet  | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free   | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor_anchors | True              | 3           | None         | presets       | None       | offset=0.5      |

    ```sh
    # Default EfficientDetD0
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py
    # Default EfficientDetD0 using input_shape 512, optimizer adamw, freezing backbone 16 epochs, total 50 + 5 epochs
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py -i 512 -p adamw --freeze_backbone_epochs 16 --lr_decay_steps 50

    # EfficientNetV2B0 backbone + EfficientDetD0 detection header
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone efficientnet.EfficientNetV2B0 --det_header efficientdet.EfficientDetD0
    # ResNest50 backbone + EfficientDetD0 header using yolox like anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone resnest.ResNest50 --use_anchor_free_mode
    # ConvNeXtTiny backbone + EfficientDetD0 header using yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone uniformer.UniformerSmall32 --use_yolor_anchors_mode

    # Typical YOLOXS with anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS --use_anchor_free_mode
    # YOLOXS with efficientdet anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS
    # ConvNeXtTiny backbone + YOLOX header with yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone coatnet.CoAtNet0 --det_header yolox.YOLOX --use_yolor_anchors_mode

    # Typical YOLOR_P6 with yolor anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolor.YOLOR_P6 --use_yolor_anchors_mode
    # YOLOR_P6 with anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolor.YOLOR_P6 --use_anchor_free_mode
    # ConvNeXtTiny backbone + YOLOR header with efficientdet anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone convnext.ConvNeXtTiny --det_header yolor.YOLOR
    ```
    **Note: COCO training still under testing, may change parameters and default behaviors. Take the risk if would like help developing.**
## Evaluation
  - **`coco_eval_script.py`** is used for evaluating model AP / AR on COCO validation set. It has a dependency `pip install pycocotools` which is not in package requirements..
    ```sh
    # resize method for EfficientDetD0 is bilinear w/o antialias
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m efficientdet.EfficientDetD0 --resize_method bilinear --disable_antialias
    # Specify --use_anchor_free_mode for YOLOX, and BGR input format
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolox.YOLOXTiny --use_anchor_free_mode --use_bgr_input --nms_method hard --nms_iou_or_sigma 0.65
    # Specify --use_yolor_anchors_mode for YOLOR.
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolor.YOLOR_CSP --use_yolor_anchors_mode --nms_method hard --nms_iou_or_sigma 0.65 \
    --nms_max_output_size 300 --nms_topk -1 --letterbox_pad 64 --input_shape 704

    # Specify h5 model
    CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m checkpoints/yoloxtiny_yolor_anchor.h5 --use_yolor_anchors_mode
    ```
  - **Tricks for evaluation from EfficientDet**
    ```py
    from keras_cv_attention_models.coco import eval_func
    from keras_cv_attention_models import efficientdet
    mm = efficientdet.EfficientDetD0()
    eval_func.run_coco_evaluation(mm, nms_score_threshold=0.001, nms_method="gaussian", nms_mode="per_class", nms_topk=5000, batch_size=8)
    ```
    | nms_score_threshold    | clip_bbox | nms_method | nms_mode  | nms_topk | Val AP 0.50:0.95, area=all |
    | ---------------------- | --------- | ---------- | --------- | -------- | -------------------------- |
    | 0.1                    | False     | hard       | global    | 0        | 0.326                      |
    | 0.001                  | False     | hard       | global    | 0        | 0.330                      |
    | 0.001                  | True      | hard       | global    | 0        | 0.331                      |
    | 0.001                  | True      | gaussian   | global    | 0        | 0.333                      |
    | 0.001                  | True      | gaussian   | per_class | 0        | 0.339                      |
    | 0.001                  | True      | gaussian   | per_class | 5000     | **0.343**                  |

  - **Tricks for evaluation from YOLOR**. Basic is `./coco_eval_script.py -m yolor.YOLOR_CSP --use_yolor_anchors_mode --nms_method hard --nms_iou_or_sigma 0.65`.
    | nms_max_output_size | nms_topk | letterbox_pad | input_shape | Val AP 0.50:0.95, area=all |
    | ------------------- | -------- | ------------- | ----------- | -------------------------- |
    | 100                 | 5000     | -1            | 640         | 0.488                      |
    | 300                 | 5000     | -1            | 640         | 0.489                      |
    | 300                 | -1       | -1            | 640         | 0.494                      |
    | 300                 | -1       | 0             | 640         | 0.496                      |
    | 300                 | -1       | 0             | 704         | 0.495                      |
    | 300                 | -1       | 64            | 704         | **0.500**                  |

  - **Methods compare**
    | Model          | nms_method | nms_iou_or_sigma | nms_max_output_size | nms_topk | letterbox_pad | input_shape | Val AP    |
    | -------------- | ---------- | ---------------- | ------------------- | -------- | ------------- | ----------- | --------- |
    | EfficientDetD1 | gaussian   | 0.5              | 100                 | 5000     | -1            | 640         | 0.402     |
    | EfficientDetD1 | hard       | 0.65             | 100                 | 5000     | -1            | 640         | 0.399     |
    | EfficientDetD1 | gaussian   | 0.5              | 300                 | -1       | -1            | 640         | **0.403** |
    | EfficientDetD1 | hard       | 0.65             | 300                 | -1       | -1            | 640         | 0.401     |
    | EfficientDetD1 | gaussian   | 0.5              | 300                 | -1       | 0             | 640         | 0.400     |
    | EfficientDetD1 | gaussian   | 0.5              | 300                 | -1       | 64            | 704         | 0.397     |
    |                |            |                  |                     |          |               |             |           |
    | YOLOXS         | gaussian   | 0.5              | 100                 | 5000     | -1            | 640         | 0.403     |
    | YOLOXS         | hard       | 0.65             | 100                 | 5000     | -1            | 640         | 0.404     |
    | YOLOXS         | hard       | 0.65             | 300                 | 5000     | -1            | 640         | 0.406     |
    | YOLOXS         | hard       | 0.65             | 300                 | -1       | -1            | 640         | **0.406** |
    | YOLOXS         | hard       | 0.65             | 300                 | -1       | 0             | 640         | 0.405     |
    | YOLOXS         | hard       | 0.65             | 300                 | -1       | 64            | 704         | 0.405     |
    |                |            |                  |                     |          |               |             |           |
    | YOLOR_CSP      | gaussian   | 0.5              | 100                 | 5000     | -1            | 640         | 0.486     |
    | YOLOR_CSP      | hard       | 0.65             | 100                 | 5000     | -1            | 640         | 0.488     |
    | YOLOR_CSP      | hard       | 0.65             | 300                 | -1       | 64            | 704         | **0.500** |
***
