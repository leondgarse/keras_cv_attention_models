# ___Keras YOLOX___
***

- **Not ready**
## Summary
  - Keras implementation of [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). Model weights converted from official publication.
  - [Paper 2107.08430 YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf).
## Models

  | Model     | Params | Image resolution | COCO test AP | Download |
  | --------- | ------ | ---------------- | ------------ | -------- |
  | YOLOXNano | 0.91M  | 416              | 25.8         | [yolox_nano_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_nano_coco.h5) |
  | YOLOXTiny | 5.06M  | 416              | 32.8         | [yolox_tiny_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_tiny_coco.h5) |
  | YOLOXS    | 9.0M   | 640              | 40.5         | [yolox_s_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_s_coco.h5) |
  | YOLOXM    | 25.3M  | 640              | 47.2         | [yolox_m_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_m_coco.h5) |
  | YOLOXL    | 54.2M  | 640              | 50.1         | [yolox_l_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_l_coco.h5) |
  | YOLOXX    | 99.1M  | 640              | 51.5         | [yolox_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_x_coco.h5) |
## Usage
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 640, 640, 3)).astype("float32")

  """ PyTorch yolox_s """
  sys.path.append('../YOLOX/')
  from exps.default import yolox_s
  import torch
  yolo_model = yolox_s.Exp()
  torch_model = yolo_model.get_model()
  _ = torch_model.eval()
  weight = torch.load('yolox_s.pth', map_location=torch.device('cpu'))["model"]
  torch_model.load_state_dict(weight)
  torch_model.head.decode_in_inference = False
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras YOLOXS """
  from keras_cv_attention_models.yolox import yolox
  mm = yolox.YOLOXS(pretrained="coco")
  keras_out = mm(inputs)
  keras_out = tf.concat([tf.reshape(ii, [-1, ii.shape[1] * ii.shape[2], ii.shape[3]]) for ii in keras_out], axis=1).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
