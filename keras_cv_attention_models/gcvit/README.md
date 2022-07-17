# ___Keras SwinTransformerV2___
***

## Summary
  - Keras implementation of [Github microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer). Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
  - Model weights reloaded from [Github microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer). `SwinTransformerV2Tiny_ns` and `SwinTransformerV2Small_ns` ported from [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py).
***

## Models
  | Model                                | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------------------ | ------ | ------ | ----- | -------- | -------- |
  | SwinTransformerV2Tiny_ns             | 28.3M  | 4.69G  | 224   | 81.8     | [tiny_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_ns_224_imagenet.h5) |
  | SwinTransformerV2Small_ns            | 49.7M  | 9.12G  | 224   | 83.5     | [small_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_ns_224_imagenet.h5) |
  |                                      |        |        |       |          |          |
  | SwinTransformerV2Tiny_window8        | 28.3M  | 5.99G  | 256   | 81.8     | [tiny_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window8_256_imagenet.h5) |
  | SwinTransformerV2Tiny_window16       | 28.3M  | 6.75G  | 256   | 82.8     | [tiny_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_window16_256_imagenet.h5) |
  | SwinTransformerV2Small_window8       | 49.7M  | 11.63G | 256   | 83.7     | [small_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window8_256_imagenet.h5) |
  | SwinTransformerV2Small_window16      | 49.7M  | 12.93G | 256   | 84.1     | [small_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_window16_256_imagenet.h5) |
  | SwinTransformerV2Base_window8        | 87.9M  | 20.44G | 256   | 84.2     | [base_window8_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window8_256_imagenet.h5) |
  | SwinTransformerV2Base_window16       | 87.9M  | 22.17G | 256   | 84.6     | [base_window16_256.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet.h5) |
  | SwinTransformerV2Base_window16, 22k  | 87.9M  | 22.17G | 256   | 86.2     | [base_window16_256_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window16_256_imagenet22k.h5) |
  | SwinTransformerV2Base_window24, 22k  | 87.9M  | 55.89G | 384   | 87.1     | [base_window24_384_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_base_window24_384_imagenet22k.h5) |
  | SwinTransformerV2Large_window16, 22k | 196.7M | 48.03G | 256   | 86.9     | [large_window16_256_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window16_256_imagenet22k.h5) |
  | SwinTransformerV2Large_window24, 22k | 196.7M | 117.1G | 384   | 87.6     | [large_window24_384_22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_large_window24_384_imagenet22k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import swin_transformer_v2

  # Will download and load pretrained imagenet weights.
  mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.77100605), ('n02123159', 'tiger_cat', 0.04094378), ...]
  ```
  **Change input resolution**. Note if `input_shape` is not divisible by `32 * window_size`, will pad for `shifted_window_attention`.
  ```py
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(input_shape=(510, 255, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_ns_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.5416627), ('n02123159', 'tiger_cat', 0.17523797), ...]
  ```
  Seems `pos_scale` and `window_size` are designed for reloading or fine-tuning with new `input_shape`. Not sure about the usage though.
  ```sh
  CUDA_VISIBLE_DEVICES='0' ./eval_script.py -m swin_transformer_v2.SwinTransformerV2Tiny_window8 -i 128 --additional_model_kwargs '{"window_size": 4, "pos_scale": 8}'
  ```
  | model                         | input_shape | window_size | pos_scale | Eval acc                    |
  | ----------------------------- | ----------- | ----------- | --------- | --------------------------- |
  | SwinTransformerV2Tiny_window8 | 128         | 8           | -1        | top1: 0.71194 top5: 0.89518 |
  | SwinTransformerV2Tiny_window8 | 128         | 4           | -1        | top1: 0.5123 top5: 0.74596  |
  | SwinTransformerV2Tiny_window8 | 128         | 4           | 8         | top1: 0.68332 top5: 0.88204 |
## Training
  - **SwinTransformerV2Tiny_ns Training** Using `A3` recipe with `batch_size=128, input_shape=(160, 160), epochs=105`. Then run `69` fine-tune epochs with `input_shape` 224. This result just showing `AdamW` works better than `LAMB` for `SwinTransformerV2`. More tuning on hyper-parameters like `mixup_alpha` or more epochs may produce a better result.
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py -m swin_transformer_v2.SwinTransformerV2Tiny_ns \
    --batch_size 128 -i 160 --lr_base_512 0.003 -p adamw -s SwinTransformerV2Tiny_ns_160_adamw_003
    ```
  - **Evaluate using input resolution `224`**:
    ```sh
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m swin_transformer_v2.SwinTransformerV2Tiny_ns -i 224 \
    --pretrained checkpoints/SwinTransformerV2Tiny_ns_160_adamw_003_latest.h5
    ```
    | Optimizer | lr_base_512 | Train acc | Best eval loss, acc on 160  | Epoch 105 eval acc on 224   |
    | --------- | ----------- | --------- | --------------------------- | --------------------------- |
    | lamb      | 0.008       | 0.6670    | Epoch 100, 0.001381, 0.7776 | top1: 0.78434 top5: 0.93982 |
    | adamw     | 0.008       | 0.6625    | Epoch 105, 0.001319, 0.7803 | top1: 0.7913 top5: 0.94458  |
    | adamw     | 0.003       | 0.6878    | Epoch 103, 0.001367, 0.7851 | top1: 0.79492 top5: 0.944   |
  - **Fine-tune 224**
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py -m swin_transformer_v2.SwinTransformerV2Tiny_ns \
    --batch_size 128 -i 224 --lr_decay_steps 64 --lr_warmup_steps 0 --lr_base_512 0.0015 -p adamw \
    --additional_model_kwargs '{"drop_connect_rate": 0.05}' --magnitude 15 \
    --pretrained checkpoints/SwinTransformerV2Tiny_ns_160_adamw_003_latest.h5 -s _drc005_E69
    ```
  - **Plot**

    ![swinv2_tiny_ns](https://user-images.githubusercontent.com/5744524/168971166-273cf560-210a-4f32-a1a5-98791d96e25e.png)
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 256, 256, 3)).astype("float32")

  """ PyTorch swinv2_tiny_window8_256 """
  sys.path.append("../pytorch-image-models")
  import timm
  import torch
  torch_model = timm.models.swinv2_tiny_window8_256(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras SwinTransformerV2Tiny_window8 """
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_window8(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
