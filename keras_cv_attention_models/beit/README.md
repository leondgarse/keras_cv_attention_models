# ___Keras BEIT___
***

## Summary
  - Keras implementation of [beit](https://github.com/microsoft/unilm/tree/master/beit). Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).
  - Model weights reloaded from [Github microsoft/beit](https://github.com/microsoft/unilm/tree/master/beit).
***

## Models
  | Model                 | Params  | Image resolution | Top1 Acc | Download                         |
  | --------------------- | ------- | ---------------- | -------- | -------------------------------- |
  | BeitBasePatch16, 21k  | 86.53M  | 224              | 85.240   | [beit_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_224.h5)  |
  |                       | 86.74M  | 384              | 86.808   | [beit_base_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_384.h5)  |
  | BeitLargePatch16, 21k | 304.43M | 224              | 87.476   | [beit_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_224.h5) |
  |                       | 305.00M | 384              | 88.382   | [beit_large_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_384.h5) |
  |                       | 305.67M | 512              | 88.584   | [beit_large_patch16_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_512.h5) |
## Usage
  ```py
  from keras_cv_attention_models import beit

  # Will download and load pretrained imagenet weights.
  mm = beit.BeitBasePatch16(input_shape=(384, 384, 3), pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7306834),
  #  ('n02123159', 'tiger_cat', 0.045104492),
  #  ('n02123045', 'tabby', 0.030001672),
  #  ('n02127052', 'lynx', 0.013072581),
  #  ('n02123597', 'Siamese_cat', 0.0062989206)]
  ```
  **Change input resolution** if input_shape is not within pre-trained, will load `MultiHeadRelativePositionalEmbedding` weights by `load_resized_pos_emb`.
  ```py
  from keras_cv_attention_models import beit
  mm = beit.BeitLargePatch16(input_shape=(640, 640, 3), num_classes=1000, pretrained="imagenet")
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/beit_large_patch16_512.h5
  # WARNING:tensorflow:Skipping loading of weights for layer block0_attn_pos_emb due to mismatch in shape ((6244, 16) vs (3972, 16)).
  # ...
  # WARNING:tensorflow:Skipping loading of weights for layer block23_attn_pos_emb due to mismatch in shape ((6244, 16) vs (3972, 16)).
  # >>>> Reload mismatched PositionalEmbedding weights: 512 -> 640
  # >>>> Reload layer: block0_attn_pos_emb
  # ...
  # >>>> Reload layer: block23_attn_pos_emb

  # Run prediction on Chelsea with (640, 640) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.53013486), ('n02123045', 'tabby', 0.18153024), ...]
  ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch beit_base_patch16_224 """
  import torch
  import timm
  torch_model = timm.models.beit_base_patch16_224(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras BotNet26T """
  from keras_cv_attention_models import beit
  mm = beit.BeitBasePatch16(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
