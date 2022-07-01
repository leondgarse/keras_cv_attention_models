# ___Keras CMT___
***

## Summary
  - CMT article: [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf).
  - `CMTTiny_torch` / `CMTXS_torch` / `CMTSmall_torch` / `CMTBase_torch` are model structure from [Github ggjy/CMT.pytorch](https://github.com/ggjy/CMT.pytorch), model weights also ported from there.
  - The main difference from `CMTTiny` and `CMTTiny_torch` is `CMTTiny` using individual `MultiHeadRelativePositionalEmbedding` in each attention block, while `CMTTiny_torch` and other `_torch` models using shared `BiasPositionalEmbedding` in attention blocks of same stack.

  ![](https://user-images.githubusercontent.com/5744524/151656779-6e6f2203-a7f7-42cf-8833-f4d472c171ae.png)
***

## Usage
  ```py
  from keras_cv_attention_models import cmt

  mm = cmt.CMTTiny()

  # Run prediction
  import tensorflow as tf
  from skimage.data import chelsea
  imm = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.99695766), ('n02123159', 'tiger_cat', 0.0021697779), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import cmt

  mm = cmt.CMTSmall_torch(input_shape=(117, 192, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/cmt_small_torch_224_imagenet.h5
  # >>>> Reload mismatched weights: 224 -> (117, 192)

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.78312486), ('n02123159', 'tiger_cat', 0.035778664), ...]
  ```
## Models
  | Model                              | Params | FLOPs | Input | Top1 Acc | Download |
  | ---------------------------------- | ------ | ----- | ----- | -------- | -------- |
  | CMTTiny, (Self trained 105 epochs) | 9.5M   | 0.65G | 160   | 77.4     |          |
  | - 305 epochs                       | 9.5M   | 0.65G | 160   | 78.94    | [cmt_tiny_160_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_160_imagenet.h5) |
  | - fine-tuned 224 (69 epochs)       | 9.5M   | 1.32G | 224   | 80.73    | [cmt_tiny_224_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_224_imagenet.h5) |
  | CMTTiny_torch, 1000 epochs         | 9.5M   | 0.65G | 160   | 79.2     | [cmt_tiny_torch_160](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_torch_160_imagenet.h5) |
  | CMTXS_torch                        | 15.2M  | 1.58G | 192   | 81.8     | [cmt_xs_torch_192](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_xs_torch_192_imagenet.h5) |
  | CMTSmall_torch                     | 25.1M  | 4.09G | 224   | 83.5     | [cmt_small_torch_224](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_small_torch_224_imagenet.h5) |
  | CMTBase_torch                      | 45.7M  | 9.42G | 256   | 84.5     | [cmt_base_torch_256](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_base_torch_256_imagenet.h5) |
## Training
  - **CMTTiny Training** Using `A3` recipe with `batch_size=256, input_shape=(160, 160), epochs=105`.
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_GPU_ALLOCATOR='cuda_malloc_async' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py \
    --seed 0 -m cmt.CMTTiny --batch_size 256 -s cmt.CMTTiny_160
    ```
  - **Evaluate using input resolution `224`**:
    ```sh
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m cmt.CMTTiny --pretrained checkpoints/cmt.CMTTiny_160_latest.h5 -i 224
    ```
    | lmhsa attention block         | Train acc | Best eval loss, acc on 160  | Epoch 105 eval acc on 224   |
    | ----------------------------- | --------- | --------------------------- | --------------------------- |
    | dw+ln, KV [dim, head, 2]      | 0.6380    | Epoch 105, 0.001398, 0.7744 | top1: 0.78766 top5: 0.94308 |
    | avg pool, KV [dim, head, 2]   | 0.6344    | Epoch 103, 0.001424, 0.7713 | top1: 0.78512 top5: 0.94194 |
    | dw+ln, KV [split2, head, dim] | 0.6350    | Epoch 103, 0.001416, 0.7719 | top1: 0.78502 top5: 0.94176 |

  - **305 epochs**:
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_GPU_ALLOCATOR='cuda_malloc_async' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py \
    --seed 0 -m cmt.CMTTiny --lr_decay_steps 300 \
    --magnitude 7 --additional_model_kwargs '{"dropout": 0.1}' -b 160 -s cmt.CMTTiny_160_E305
    ```
    | 305 epochs    | additional_model_kwargs | Train acc | Best eval loss, acc on 160 | Epoch 305 eval acc on 224   |
    | ------------- | ----------------------- | --------- | -------------------------- | --------------------------- |
    | mag6, bs 256  |                         | 0.6702    | Epoch 304, 0.0013, 0.7874  | top1: 0.79956 top5: 0.94850 |
    | mag15, bs 256 |                         | 0.6390    | Epoch 304, 0.0014, 0.7824  | top1: 0.79630 top5: 0.94794 |
    | mag7, bs 160  | drop_connect_rate 0.05  | 0.6577    | Epoch 294, 0.0013, 0.7880  | top1: 0.80126 top5: 0.94898 |
    | mag7, bs 160  | dropout 0.1             | 0.6655    | Epoch 296, 0.0013, 0.7894  | top1: 0.80136 top5: 0.94954 |

  - **Plot** 305 epochs ones are plotted every 3 epochs

    ![cmt_tiny](https://user-images.githubusercontent.com/5744524/167232239-87105c93-799d-48d0-8773-a3e5af0e29c4.png)
  - **Fine-tune 224**. Note without fine-tune 224 accuracy is `0.80136`, just improved to `0.8073` by this, not a big one.
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 -m cmt.CMTTiny \
    --pretrained checkpoints/cmt.CMTTiny_160_E305_latest.h5 -i 224 --batch_size 64 \
    --lr_decay_steps 64 --lr_warmup_steps 0 --lr_base_512 0.004 \
    --additional_model_kwargs '{"drop_connect_rate": 0.05}' --magnitude 8 -s _drc_005
    ```
    ![cmt_tiny_ft_224](https://user-images.githubusercontent.com/5744524/167232247-04a7ed70-61ea-4316-9af2-58ce27efb8b5.png)
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch cmt_s """
  sys.path.append('../pytorch-image-models/')
  sys.path.append('../Efficient-AI-Backbones/')

  from cmt_pytorch import cmt as cmt_pytorch
  import torch
  torch_model = cmt_pytorch.cmt_s(img_size=224)
  _ = torch_model.eval()
  weight = torch.load('cmt_small.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(weight['model'])
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras CMTSmall_torch """
  from keras_cv_attention_models import cmt
  mm = cmt.CMTSmall_torch(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
