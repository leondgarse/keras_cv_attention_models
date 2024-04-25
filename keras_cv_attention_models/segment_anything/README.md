# ___Keras Segment Anything___
***

## Summary
  - Paper [PDF 2304.02643 Segment Anything](https://arxiv.org/abs/2304.02643)
  - Paper [PDF 2306.14289 FASTER SEGMENT ANYTHING: TOWARDS LIGHTWEIGHT SAM FOR MOBILE APPLICATIONS](https://arxiv.org/pdf/2306.14289.pdf)
  - Paper [PDF 2312.13789 TinySAM: Pushing the Envelope for Efficient Segment Anything Mode](https://arxiv.org/pdf/2312.13789.pdf)
  - [Github facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
  - [Github ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - [Github xinghaochen/TinySAM](https://github.com/xinghaochen/TinySAM)
  - MobileSAM weights ported from [Github ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - EfficientViT_SAM weights ported from [Github mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
## Models
  | Model               | Params | FLOPs | Input | COCO val mask AP | Download |
  | ------------------- | ------ | ----- | ----- | ---------------- | -------- |
  | MobileSAM           | 5.74M  | 39.4G | 1024  | 41.0             | [mobile_sam_5m_image_encoder](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/mobile_sam_5m_image_encoder_1024_sam.h5)  |
  | TinySAM             | 5.74M  | 39.4G | 1024  | 41.9             | [tinysam_5m_image_encoder](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/tinysam_5m_image_encoder_1024_sam.h5)     |
  | EfficientViT_SAM_L0 | 30.73M | 35.4G | 512   | 45.7             | [efficientvit_sam_l0_image_encoder](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/efficientvit_sam_l0_image_encoder_1024_sam.h5)  |

  Model differences only in `ImageEncoder`, the SAM `PromptEncoder` and `MaskDecoder` are sharing the same one

  | Model                   | Params | FLOPs | Download |
  | ----------------------- | ------ | ----- | -------- |
  | MaskDecoder             | 4.06M  | 1.78G | [sam_mask_decoder_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_mask_decoder_sam.h5)         |
  | - tiny_sam              | 4.06M  | 1.78G | [tiny_sam_mask_decoder_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/tiny_sam_mask_decoder_sam.h5)    |
  | PointsEncoder           | 768    | 0     | [sam_prompt_encoder_points_encoder_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_prompt_encoder_points_encoder_sam.h5) |
  | BboxesEncoder           | 512    | 256   | [sam_prompt_encoder_bboxes_encoder_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_prompt_encoder_bboxes_encoder_sam.h5) |
  | MaskEncoder             | 4684   | 0     | [sam_prompt_encoder_mask_encoder_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_prompt_encoder_mask_encoder_sam.h5) |
  | EmptyMask               | 256    | 0     | [sam_prompt_encoder_empty_mask_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_prompt_encoder_empty_mask_sam.h5) |
  | PositionEmbeddingRandom | 256    | 0     | [sam_prompt_encoder_positional_embedding_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/segment_anything/sam_prompt_encoder_positional_embedding_sam.h5) |
## Usage
  - **Basic**
    ```py
    from keras_cv_attention_models import segment_anything, test_images
    mm = segment_anything.MobileSAM()
    image = test_images.dog_cat()
    masks, iou_predictions, low_res_masks = mm(image)
    fig = mm.show(image, masks, iou_predictions, save_path='aa.jpg')
    ```
    ![sam_mobile_sam_5m_raw](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/da678689-e613-4b04-8f65-f834e565b504)
  - **Call args**
    - **`points`**: combinging with `labels`, specific points coordinates as background or foreground. `np.array` value in shape `[None, 2]`, where `2` means `[left, top]`. left value range in `[0, 1]` or `[0, width]`, and top in `[0, 1]` or `[0, height]`.
    - **`labels`**: combinging with `points`, specific points coordinates as background or foreground. `np.array` value in shape `[None]`, value in `[0, 1]`, where 0 means relative point being background, and 1 foreground.
    - **`boxes`**: specific box area performing segmentation. np.array value in shape `[1, 4]`, where `4` means `[left, top, right, bottom]`. left and right value range in `[0, 1]` or `[0, width]`, and top and bottom in `[0, 1]` or `[0, height]`. Supports only single boxes as inputs.
    - **`masks`**: specific masks area performing segmentation. np.array value in shape `[height, width]`, where `height` and `width` should better be `256`, or will perform nearest resize on it.
  - **Outputs**
    - **`masks`** is all masks output, and it's `4` masks by default, specified by `MaskDecoder` parameter `num_mask_tokens`. Default shape is `[4, image_height, image_width]`. **`masks[0]`** is the output of token 0, which is said better for using if segmenting **single object with multi prompts**. **`masks[1:]`** are intended for **ambiguous input prompts**, and **`iou_predictions[1:]`** are the corresponding confidences, which can be used for picking the highest score one from `masks[1:]`.
    - **`iou_predictions`** is the corresponding masks confidences. Default shape is `[4]`.
    - **`low_res_masks`** is the raw output from `MaskDecoder`. Default shape is `[4, 256, 256]`.
  - **Specific a point and label as foreground**
    ```py
    from keras_cv_attention_models import segment_anything, test_images
    mm = segment_anything.MobileSAM()
    image = test_images.dog_cat()
    points, labels = np.array([(400, 256)]), np.array([1])
    masks, iou_predictions, low_res_masks = mm(image, points, labels)
    fig = mm.show(image, masks, iou_predictions, points=points, labels=labels, save_path='bb.jpg')
    ```
    ![sam_mobile_sam_5m](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/b4d5dbc7-69d9-47b1-936b-64bd00e7ec3e)
  - **Specific a point and label as foreground, and also a box area**
    ```py
    from keras_cv_attention_models import segment_anything, test_images
    mm = segment_anything.EfficientViT_SAM_L0()
    image = test_images.dog_cat()
    points, labels, boxes = [[0.8, 0.8]], [1], [0.5, 0.5, 1, 1]
    masks, iou_predictions, low_res_masks = mm(image, points, labels, boxes)
    fig = mm.show(image, masks, iou_predictions, points=points, labels=labels, boxes=boxes, save_path='cc.jpg')
    ```
    ![sam_efficientvit_l0_box](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/45c94413-d0b9-4ced-b1c5-83efb15634e1)
  - **Cooperate with detection model** **Note: bbox area required is a single one in format `[left, top, right, bottom]`, while detection models in this repo output bbox area in format `[top, left, bottom, right]`**.
    ```py
    from keras_cv_attention_models import segment_anything, test_images, yolov8
    det = yolov8.YOLOV8_N()
    mm = segment_anything.EfficientViT_SAM_L0()

    image = test_images.dog_cat()
    boxes, _, _ = det.decode_predictions(det(det.preprocess_input(image)))[0]
    boxes = np.array(boxes)[0, [1, 0, 3, 2]]  # Pick a single one, and [top, left, bottom, right] -> [left, top, right, bottom]
    masks, iou_predictions, low_res_masks = mm(image, boxes=boxes)
    fig = mm.show(image, masks, iou_predictions, boxes=boxes, save_path='dd.jpg')
    ```
    ![sam_efficientvit_l0_yolov8n_box](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/c393a5b7-1849-4a27-ba68-221c36321316)
  - **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
    ```py
    os.environ['KECAM_BACKEND'] = 'torch'
    from keras_cv_attention_models import segment_anything, test_images
    # >>>> Using PyTorch backend
    mm = segment_anything.EfficientViT_SAM_L0()
    image = test_images.dog_cat()
    points, labels = [[0.5, 0.8], [0.5, 0.2], [0.8, 0.8]], [1, 1, 0]
    masks, iou_predictions, low_res_masks = mm(image, points, labels)
    fig = mm.show(image, masks, iou_predictions, points=points, labels=labels, save_path='ee.jpg')
    ```
    ![sam_efficientvit_l0](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/72135535-1bfe-4ab0-abe6-980ce50c8045)
## Verification with PyTorch version
  ```py
  """ PyTorch MobileSAM """
  sys.path.append("../pytorch-image-models/")
  sys.path.append('../MobileSAM/')
  from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

  mobile_sam = sam_model_registry["vit_t"](checkpoint="../MobileSAM/weights/mobile_sam.pt")
  _ = mobile_sam.eval()
  predictor = SamPredictor(mobile_sam)

  from PIL import Image
  from keras_cv_attention_models import test_images
  # Resize ahead, as torch one using BILINEAR, and kecam using BICUBIC
  image = np.array(Image.fromarray(test_images.dog_cat()).resize([1024, 1024], resample=Image.Resampling.BILINEAR))
  point_coords, point_labels = np.array([(400, 400)]), np.array([1])
  predictor.set_image(image)
  torch_out = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

  """ Kecam MobileSAM """
  from keras_cv_attention_models import segment_anything
  mm = segment_anything.MobileSAM()
  masks, iou_predictions, low_res_masks = mm(image, point_coords, point_labels)

  """ Verification """
  print(f"{np.allclose(torch_out[0], masks[1:, :, :]) = }")
  # np.allclose(torch_out[0], masks[1:, :, :]) = True
  print(f"{torch_out[1] = }")
  # torch_out[1] = array([0.8689907 , 0.7555798 , 0.99140215], dtype=float32)
  print(f"{iou_predictions[1:] = }")
  # iou_predictions[1:] = array([0.868991  , 0.7555795 , 0.99140203], dtype=float32)
  print(f"{np.allclose(torch_out[2], low_res_masks[1:, :, :], atol=1e-4) = }")
  # np.allclose(torch_out[2], low_res_masks[1:, :, :], atol=1e-4) = True
  ```
  **EfficientViT-L0-SAM**
  ```py
  """ PyTorch EfficientViT-L0-SAM """
  sys.path.append("../pytorch-image-models/")
  sys.path.append('../efficientvit/')
  import torch
  from efficientvit.sam_model_zoo import create_sam_model
  from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
  tt = create_sam_model('l0', weight_url='EfficientViT-L0-SAM.pt')
  _ = tt.eval()
  efficientvit_sam_predictor = EfficientViTSamPredictor(tt)

  os.environ['KECAM_BACKEND'] = 'torch'  # Using Torch backend here, TF bicubic resize is different from Torch, allclose atol could be rather high
  from keras_cv_attention_models import test_images
  point_coords, point_labels = np.array([(256, 256)]), np.array([1])

  image = test_images.dog_cat()
  efficientvit_sam_predictor.set_image(image)
  torch_out = efficientvit_sam_predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

  """ Kecam EfficientViT_SAM_L0 with PyTorch backend """
  from keras_cv_attention_models import segment_anything
  # >>>> Using PyTorch backend
  mm = segment_anything.EfficientViT_SAM_L0()
  masks, iou_predictions, low_res_masks = mm(image, point_coords, point_labels)

  """ Verification """
  same_masks = (torch_out[0] == masks[1:, :, :]).sum() / np.prod(torch_out[0].shape)
  print("same masks percentage: {:.6f}%".format(same_masks * 100))
  # same masks percentage: 99.999619%
  print(f"{torch_out[1] = }")
  # torch_out[1] = array([0.6856826 , 0.998912  , 0.96785474], dtype=float32)
  print(f"{iou_predictions[1:] = }")
  # iou_predictions[1:] = array([0.68567175, 0.99891114, 0.96785533], dtype=float32)
  print(f"{np.allclose(torch_out[2], low_res_masks[1:, :, :], atol=1e-3) = }")
  # np.allclose(torch_out[2], low_res_masks[1:, :, :], atol=1e-3) = True
  ```
