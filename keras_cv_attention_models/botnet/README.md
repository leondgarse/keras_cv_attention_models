# ___Keras BotNet___
***

## Summary
  - Keras implementation of [botnet](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2). Paper [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).
  - Model weights reloaded from timm [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
***

## Models
  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | BotNet50      | 21M    | 5.42G  | 224   |          |          |
  | BotNet101     | 41M    | 9.13G  | 224   |          |          |
  | BotNet152     | 56M    | 12.84G | 224   |          |          |
  | BotNet26T     | 12.5M  | 3.30G  | 256   | 79.246   | [botnet26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet26t_256_imagenet.h5) |
  | BotNextECA26T | 10.59M | 2.45G  | 256   | 79.270   | [botnext_eca26t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnext_eca26t_256_imagenet.h5) |
  | BotNetSE33T   | 13.7M  | 3.89G  | 256   | 81.2     | [botnet_se33t_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet_se33t_256_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import botnet

  # Will download and load pretrained imagenet weights.
  # Only BotNet50 weights supported, BotNet101 / BotNet152 will be random inited.
  mm = botnet.BotNet26T(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7356877),
  #  ('n02123045', 'tabby', 0.057901755),
  #  ('n02123159', 'tiger_cat', 0.040991902),
  #  ('n02127052', 'lynx', 0.0043538176),
  #  ('n02123597', 'Siamese_cat', 0.0007328492)]
  ```
  **Change input resolution** if input_shape is not `(224, 224, 3)`, will load `PositionalEmbedding` weights by `load_resized_weights`.
  ```py
  from keras_cv_attention_models import botnet
  mm = botnet.BotNet26T(input_shape=(320, 320, 3), num_classes=1000, pretrained="imagenet")
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/botnet26t_256_imagenet.h5
  # WARNING:tensorflow:Skipping loading of weights for layer stack3_block2_deep_2_mhsa_pos_emb due to mismatch in shape ((64, 39) vs (64, 31)).
  # WARNING:tensorflow:Skipping loading of weights for layer stack3_block2_deep_2_mhsa_pos_emb due to mismatch in shape ((64, 39) vs (64, 31)).
  # WARNING:tensorflow:Skipping loading of weights for layer stack4_block1_deep_2_mhsa_pos_emb due to mismatch in shape ((128, 39) vs (128, 31)).
  # WARNING:tensorflow:Skipping loading of weights for layer stack4_block1_deep_2_mhsa_pos_emb due to mismatch in shape ((128, 39) vs (128, 31)).
  # WARNING:tensorflow:Skipping loading of weights for layer stack4_block2_deep_2_mhsa_pos_emb due to mismatch in shape ((128, 19) vs (128, 15)).
  # WARNING:tensorflow:Skipping loading of weights for layer stack4_block2_deep_2_mhsa_pos_emb due to mismatch in shape ((128, 19) vs (128, 15)).
  # >>>> Reload mismatched PositionalEmbedding weights: 256 -> 320
  # >>>> Reload layer: stack3_block2_deep_2_mhsa_pos_emb
  # >>>> Reload layer: stack4_block1_deep_2_mhsa_pos_emb
  # >>>> Reload layer: stack4_block2_deep_2_mhsa_pos_emb

  # Run prediction on Chelsea with (320, 320) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.6976793), ('n02123045', 'tabby', 0.055820007), ...]
  ```
## Relative to absolute positional embedding detail
  - [How Positional Embeddings work in Self-Attention (code in Pytorch)](https://theaisummer.com/positional-embeddings/)
  - For the first row, length is `2 * width - 1`, elements wanted to keep is the last `width`. So elements cut in the first row is `2 * width - 1 - width == width - 1`.
  - Then **flatten** the last two dimensions, and the number of `cutout elements` between `keep elements` is fixed as `width - 2`.
  - The **reshape** target of the flattened one is `(width, width + width - 2) --> (width, 2 * (width - 1))`
  - Keep the head `width` elements in each row for **output**.
  ```py
  rel_pos = np.arange(28).reshape(1, 1, 1, 4, 7) # [bs, heads, height, width, 2 * width - 1]
  print(rel_pos[0, 0, 0])
  # [[ 0,  1,  2,  3,  4,  5,  6],
  #  [ 7,  8,  9, 10, 11, 12, 13],
  #  [14, 15, 16, 17, 18, 19, 20],
  #  [21, 22, 23, 24, 25, 26, 27]]
  _, heads, hh, ww, dim = rel_pos.shape

  # (ww, 2 * ww - 1) --> (ww, 2 * (ww - 1)) ==> removed: ww * (2 * ww - 1) - ww * 2 * (ww - 1) == ww
  flat_x = rel_pos.reshape([-1, heads, hh, ww * (ww * 2 - 1)])
  print(flat_x[0, 0])
  # [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
  flat_x = flat_x[:, :, :, ww - 1:-1]
  print(flat_x[0, 0])
  # [[ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
  final_x = flat_x.reshape([-1, heads, hh, ww, 2 * (ww - 1)])
  print(final_x[0, 0])
  # [[[ 3,  4,  5,  6,  7,  8],
  #   [ 9, 10, 11, 12, 13, 14],
  #   [15, 16, 17, 18, 19, 20],
  #   [21, 22, 23, 24, 25, 26]]]
  final_x = final_x[:, :, :, :, :ww]
  print(final_x[0, 0])
  # [[[ 3,  4,  5,  6],
  #   [ 9, 10, 11, 12],
  #   [15, 16, 17, 18],
  #   [21, 22, 23, 24]]]
  ```
  ![](https://user-images.githubusercontent.com/5744524/151656818-ef730fc4-d355-4964-b837-d5fbd28b87ac.png)
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 256, 256, 3)).astype("float32")

  """ PyTorch botnet26t_256 """
  import torch
  import timm
  torch_model = timm.models.botnet26t_256(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras BotNet26T """
  from keras_cv_attention_models import botnet
  mm = botnet.BotNet26T(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
