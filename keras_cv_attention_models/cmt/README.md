# ___Keras CMT___
***

## Summary
  - CMT article: [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)
  - [Github wilile26811249/CMT_CNN-meet-Vision-Transformer](https://github.com/wilile26811249/CMT_CNN-meet-Vision-Transformer)

  ![](https://user-images.githubusercontent.com/5744524/151656779-6e6f2203-a7f7-42cf-8833-f4d472c171ae.png)
***

## Usage
  ```py
  from keras_cv_attention_models import cmt

  # Only CMTTiny pretrained available.
  mm = cmt.CMTTiny()

  # Run prediction
  import tensorflow as tf
  from skimage.data import chelsea
  imm = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.9962525), ('n02123159', 'tiger_cat', 0.0025533703), ...]
  ```
## Training
  - **CMTTiny Training** Using `A3` recipe with `batch_size=256, input_shape=(160, 160), epochs=105`. Note paper reported accuracy is trained `1000` epochs [Results on ImageNet #1](https://github.com/FlyEgle/CMT-pytorch/issues/1).
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
    --magnitude 7 --additional_model_kwargs '{"drop_connect_rate": 0.05}' -b 160 \
    ```
    | 305 epochs             | Train acc | Best eval loss, acc on 160 | Epoch 305 eval acc on 224   |
    | ---------------------- | --------- | -------------------------- | --------------------------- |
    | mag6, drc 0, bs 256    | 0.6702    | Epoch 304, 0.0013, 0.7874  | top1: 0.79956 top5: 0.94850 |
    | mag7, drc 0.05, bs 160 | 0.6577    | Epoch 294, 0.0013,0.7880   | top1: 0.80126 top5: 0.94898 |
    | mag15, drc 0, bs 256   | 0.6390    | Epoch 304, 0.0014,0.7824   | top1: 0.79630 top5: 0.94794 |

  - **Plot** 305 epochs ones are plotted every 3 epochs

    ![cmt_tiny](https://user-images.githubusercontent.com/5744524/159236868-06d7f2a3-3a17-433a-99ff-148640b25d92.png)
## Models
  | Model                              | Params | Image resolution | Top1 Acc | Download |
  | ---------------------------------- | ------ | ---------------- | -------- | -------- |
  | CMTTiny, (Self trained 105 epochs) | 9.5M   | 160              | 77.4     |          |
  | - 305 epochs                       | 9.5M   | 160              | 78.8     | [cmt_tiny_160_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cmt/cmt_tiny_160_imagenet.h5) |
  | - evaluate 224 (not fine-tuned)    | 9.5M   | 224              | 80.1     |          |
  | CMTTiny, 1000 epochs               | 9.5M   | 160              | 79.2     |          |
  | CMTXS                              | 15.2M  | 192              | 81.8     |          |
  | CMTSmall                           | 25.1M  | 224              | 83.5     |          |
  | CMTBig                             | 45.7M  | 256              | 84.5     |          |
***
