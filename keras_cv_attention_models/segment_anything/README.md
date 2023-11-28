# ___Keras Segment Anything___
***

## Summary
  - [Github facebookresearch/segment-anything(https://github.com/facebookresearch/segment-anything)
  - [Github ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - Model weights ported from [Github ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
## Models
  | Model | Params | FLOPs | Input | Download |     |
  | ----- | ------ | ----- | ----- | -------- | --- |
  |       |        |       |       |          |     |
## Usage
  - **Basic [Still not good, just working]**
    ```py
    from keras_cv_attention_models import test_images
    from keras_cv_attention_models.segment_anything import sam
    mm = sam.SAM()
    image = test_images.dog_cat()
    points, labels = np.array([[400, 400]]), np.array([1])
    masks, iou_predictions, low_res_masks = mm(image, points, labels)
    fig = mm.show(image, masks, iou_predictions, points=points, labels=labels, save_path='aa.jpg')
    ```
    ![segment_anything](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/e3013d4e-1c28-426a-bb88-66144c8413ac)
