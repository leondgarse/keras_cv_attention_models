# Keras_CMT
***

## Summary
  - CMT article: [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)
  - [Github wilile26811249/CMT_CNN-meet-Vision-Transformer](https://github.com/wilile26811249/CMT_CNN-meet-Vision-Transformer)
  - No pretrained available.

  ![](https://user-images.githubusercontent.com/5744524/151656779-6e6f2203-a7f7-42cf-8833-f4d472c171ae.png)
***

## Usage
  ```py
  from keras_cv_attention_models import cmt

  # No pretrained available.
  mm = cmt.CMTTiny()
  mm.summary()
  ```
## Training
  - **Training** Using `A3` recipe with `batch_size=256, input_shape=(160, 160), epochs=105`. Note paper reported accuracy is trained `1000` epochs [Results on ImageNet #1](https://github.com/FlyEgle/CMT-pytorch/issues/1).
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_GPU_ALLOCATOR='cuda_malloc_async' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py \
    --seed 0 -m cmt.CMTTiny --batch_size 256 -s cmt.CMTTiny_160
    ```
    **Evaluate using input resolution `224`**:
    ```sh
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m cmt.CMTTiny --pretrained checkpoints/cmt.CMTTiny_160_latest.h5 -i 224
    ```
    | lmhsa attention block         | Train acc | Best eval loss, acc on 160  | Epoch 105 Eval acc on 224   |
    | ----------------------------- | --------- | --------------------------- | --------------------------- |
    | dw+ln, KV [dim, head, 2]      | 0.6380    | Epoch 105, 0.001398, 0.7744 | top1: 0.78766 top5: 0.94308 |
    | avg pool, KV [dim, head, 2]   | 0.6344    | Epoch 103, 0.001424, 0.7713 | top1: 0.78512 top5: 0.94194 |
    | dw+ln, KV [split2, head, dim] | 0.6350    | Epoch 103, 0.001416, 0.7719 | top1: 0.78502 top5: 0.94176 |

    ![](https://user-images.githubusercontent.com/5744524/156691026-233fa5b5-b1b3-489c-a6ad-f2fa1b987cbe.png)
## Models
  | Model    | Params | Image resolution | Top1 Acc |
  | -------- | ------ | ---------------- | -------- |
  | CMTTiny  | 9.5M   | 160              | 79.2     |
  | CMTXS    | 15.2M  | 192              | 81.8     |
  | CMTSmall | 25.1M  | 224              | 83.5     |
  | CMTBig   | 45.7M  | 256              | 84.5     |
***
