# ___COCO___
***
## Data
  - Default data augment for training is `mosaic mix prob=0.5` + `randaug magnitude=6 with rotate, shear, transpose` + `random largest crop`.
    - For `mosaic_mix_prob`, it's applied per batch, means each image is repeated `4` times, then randomly shuffled and mosaic mixup with others in an entire batch.
    - For `random_crop_mode`, it controls how random crop / scale is applied, `0` for eval mode, `(0, 1)` for random crop, `1` for random largest crop, `> 1` for random scale.
    ```py
    from keras_cv_attention_models import coco
    train_dataset, test_dataset, _, _, _ = coco.init_dataset(batch_size=4, mosaic_mix_prob=0.5, random_crop_mode=1.0, magnitude=6)
    coco.show_batch_sample(train_dataset)
    ```
    ![coco_data_aug](https://user-images.githubusercontent.com/5744524/158043958-8eb20745-e83f-4dd8-8e41-b77d56224c3c.png)
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
  - Default parameters for `coco_train_script.py` is `EfficientDetD0` with `input_shape=(256, 256, 3), batch_size=64, mosaic_mix_prob=0.5, freeze_backbone_epochs=32, total_epochs=105`. Technically, it's any `pyramid structure backbone` + `EfficientDet / YOLOX header` + `anchor_free / anchors` combination supported.
    ```sh
    # Default EfficientDetD0
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py
    # Default EfficientDetD0 using input_shape 512, optimizer adamw, freezing backbone 16 epochs, total 50 + 5 epochs
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py -i 512 -p adamw --freeze_backbone_epochs 16 --lr_decay_steps 50

    # EfficientNetV2B0 backbone + EfficientDetD0 detection header
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone efficientnet.EfficientNetV2B0 --det_header efficientdet.EfficientDetD0
    # ResNest50 backbone + EfficientDetD0 header using yolox like anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone resnest.ResNest50 --use_anchor_free_mode

    # Typical YOLOXS with anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS --use_anchor_free_mode
    # YOLOXS with efficientdet like anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolox.YOLOXS
    # CoAtNet0 backbone + YOLOXS header with anchor_free_mode
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone coatnet.CoAtNet0 --det_header yolox.YOLOXS --use_anchor_free_mode
    # UniformerSmall32 backbone + YOLOX header with efficientdet like anchors
    CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --backbone uniformer.UniformerSmall32 --det_header yolox.YOLOX
    ```
## Evaluation
  - Specifying `--data_name coco` using `eval_script.py` for evaluating COCO AP. It has a dependency `pip install pycocotools` which is not in package requirements.
    ```sh
    # resize method for EfficientDetD0 is bilinear w/o antialias
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m efficientdet.EfficientDetD0 -d coco --batch_size 8 --resize_method bilinear --disable_antialias
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
    # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
    # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.366
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.132
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.460
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.484
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
    ```
    **Evaluating specific h5 model**
    ```sh
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m checkpoints/xxx.h5 -d coco --batch_size 8
    ```
  - **Tricks for evaluation**
    ```py
    from keras_cv_attention_models.coco import eval_func
    from keras_cv_attention_models import efficientdet
    mm = efficientdet.EfficientDetD0()
    eval_func.run_coco_evaluation(mm, nms_score_threshold=0.001, nms_method="gaussian", nms_mode="per_class", nms_topk=5000, batch_size=8)
    ```
    | nms_score_threshold | clip_bbox | nms_method | nms_mode  | nms_topk | Val AP 0.50:0.95, area=all |
    | ------------------- | --------- | ---------- | --------- | -------- | -------------------------- |
    | 0.1                 | False     | hard       | global    | -1       | 0.326                      |
    | 0.001               | False     | hard       | global    | -1       | 0.330                      |
    | 0.001               | True      | hard       | global    | -1       | 0.331                      |
    | 0.001               | True      | gaussian   | global    | -1       | 0.333                      |
    | 0.001               | True      | gaussian   | per_class | -1       | 0.339                      |
    | 0.001               | True      | gaussian   | per_class | 5000     | **0.343**                  |
***
