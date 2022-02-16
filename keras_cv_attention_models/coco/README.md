# ___COCO___
***

## Evaluation
  - Specifying `--data_name coco` using `eval_script.py` for evaluating COCO AP.
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
