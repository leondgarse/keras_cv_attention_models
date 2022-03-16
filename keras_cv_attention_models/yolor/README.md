# ___Keras YOLOR___
***

## Summary
  - Keras implementation of [Github WongKinYiu/yolor](https://github.com/WongKinYiu/yolor). Model weights converted from official publication.
  - [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).
  - Model weights may change, for better compiling with already existing implementations.
## Models
  | Model      | Params | Image resolution | COCO val AP | Download |
  | ---------- | ------ | ---------------- | ----------- | -------- |
  | YOLOR_CSP  | 52.9M  | 640              | 50.0        | [yolor_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_coco.h5) |
  | YOLOR_CSPX | 99.8M  | 640              | 51.5        | [yolor_csp_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_x_coco.h5) |
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import yolor
    model = yolor.YOLOR_CSP(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, is_bbox_width_first=True)
    ```
    ![yolor_csp_dog_cat](https://user-images.githubusercontent.com/5744524/158571726-30ce61ed-ff1a-47ef-9a32-bfe546497a6d.png)
