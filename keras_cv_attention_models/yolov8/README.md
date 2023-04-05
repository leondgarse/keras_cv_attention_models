# ___Keras YOLOV8___
***

## Summary
  - Keras implementation of [Github ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). Model weights converted from official publication.
***

## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models.yolov8 import yolov8
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
