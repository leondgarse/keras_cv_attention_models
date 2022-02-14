# ___COCO___
***

## Evaluation
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
  | 0.001               | True      | gaussian   | per_class | 5000     | 0.343                      |
***
