# ___ImageNet___
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Training](#training)
	- [aotnet.AotNet50](#aotnetaotnet50)
	- [Comparing resize methods bicubic or bilinear](#comparing-resize-methods-bicubic-or-bilinear)
	- [Comparing rescale mode torch or tf](#comparing-rescale-mode-torch-or-tf)
	- [Comparing optimizer LAMB or AdamW](#comparing-optimizer-lamb-or-adamw)
	- [Test EvoNormalization](#test-evonormalization)
- [Progressive training](#progressive-training)
	- [EfficientNetV2B0 cifar10 basic test](#efficientnetv2b0-cifar10-basic-test)
	- [AotNet50 A3 progressive 96 128 160](#aotnet50-a3-progressive-96-128-160)

<!-- /TOC -->
***

# Training
## aotnet.AotNet50
  - `aotnet.AotNet50` default parameters set is a typical `ResNet50` architecture with `Conv2D use_bias=False` and `padding` like `PyTorch`.
  - **A3**. Default params for `train_script.py` is like `A3` configuration from [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476.pdf) with `batch_size=256, input_shape=(160, 160)`.
    ```sh
    # `antialias` is default enabled for resize, can be turned off be set `--disable_antialias`.
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 -s aotnet50
    ```
    Evaluating on best one
    ```sh
    # Evaluation using input_shape (224, 224).
    # `antialias` usage should be same with training.
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m aotnet50_epoch_103_val_acc_0.7674.h5 -i 224 --central_crop 0.95
    # >>>> Accuracy top1: 0.78466 top5: 0.94088
    ```
  - **A2**. Train A2 recipe from [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476.pdf) with `batch_size=128, input_shape=(224, 224)`
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py \
    --seed 0 --input_shape 224 --lr_base_512 5e-3 --lr_decay_steps 300 --batch_size 128 \
    --magnitude 7 --additional_model_kwargs '{"drop_connect_rate": 0.05}' -s aotnet50_A2
    ```
    The final best result is **`top1: 0.79706 top5: 0.9461`**.
  - **Plot**. The last `AotNet50, A2, 224, Epoch 300` is plotted every `3` epochs, matching with other `A3` ones.
    ```py
    import json
    from keras_cv_attention_models.imagenet import eval_func
    hhs = {
        "timm, Resnet50, A3, 160, Epoch 100": eval_func.parse_timm_log("../pytorch-image-models/log.foo", pick_keys=['loss', 'val_acc']),
        "AotNet50, A3, 160, Epoch 100": "checkpoints/aotnet50_hist.json",
        "AotNet50, A2, 224, Epoch 100": "checkpoints/aotnet50_A2_E100_hist.json",
        "AotNet50, A2, 224, Epoch 300": {kk: vv[::3] for kk, vv in json.load(open("checkpoints/aotnet50_A2_hist.json", "r")).items()},
    }
    fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
    ```
    ![aotnet50_imagenet](https://user-images.githubusercontent.com/5744524/163795114-b2441e5d-94d5-4310-826a-958426f1343e.png)
## Comparing resize methods bicubic or bilinear
  - Basic standard is `AotNet50` + `A3` configuration from [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476.pdf) with `batch_size=256, input_shape=(160, 160)`.
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0
    ```

  | Resize method | anti alias | Train acc | Best eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
  | ------------- | ---------- | --------- | --------------------------- | -------------------------- | ------------------ |
  | bicubic       | True       | 0.6310    | Epoch 103, 0.001452, 0.7674 | 0.78466, 0.94088           | 0.78476, 0.94098   |
  | bicubic       | False      | 0.6313    | Epoch  97, 0.001481, 0.7626 | 0.77994, 0.93800           | 0.77956, 0.93808   |
  | bilinear      | True       | 0.6296    | Epoch 104, 0.001491, 0.7676 | 0.78152, 0.93924           | 0.78128, 0.93944   |
  | bilinear      | False      | 0.6310    | Epoch 103, 0.001455, 0.7642 | 0.78024, 0.93974           | 0.78072, 0.93996   |

  Thus `anti alias` is default enabled, can be turned off be specifying `--disable_antialias`.
## Comparing rescale mode torch or tf
  - Resize method using `bicubic + anti_alias`.

  | Rescale mode | Train acc | Best eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
  | ------------ | --------- | --------------------------- | -------------------------- | ------------------ |
  | torch        | 0.6310    | Epoch 103, 0.001452, 0.7674 | 0.78466, 0.94088           | 0.78476, 0.94098   |
  | tf           | 0.6328    | Epoch  97, 0.001452, 0.7671 | 0.78316, 0.93898           | 0.78310, 0.93910   |
## Comparing optimizer LAMB or AdamW
  - Using `--optimizer adamw` testing learning_rate and weight_decay.

  | lr base | Weight decay   | Train acc | Best Eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
  | ------- | -------------- | --------- | --------------------------- | -------------------------- | ------------------ |
  | 4e-3    | 0.05           | 0.6216    | Epoch 102, 0.001468, 0.7638 | 0.77862, 0.93876           | 0.77918, 0.93850   |
  | 4e-3    | 0.02           | 0.6346    | Epoch 100, 0.001471, 0.7669 | 0.78060, 0.93842           | 0.78058, 0.93856   |
  | 8e-3    | 0.02           | 0.6285    | Epoch 105, 0.001463, 0.7675 | 0.78268, 0.93828           | 0.78268, 0.93828   |

  ![aotnet_adamw](https://user-images.githubusercontent.com/5744524/152712334-478ceb8d-e22f-4b3f-a321-af7350f53a08.png)
## Test EvoNormalization
  - [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py), paper [PDF 2004.02967 Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf).
  ```sh
  CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 \
  --additional_model_kwargs '{"use_evo_norm": true, "evo_norm_group_size": 16}' \
  -s aotnet50_evonorm
  ```
  | evonorm             | Train acc | Best eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
  | ------------------- | --------- | --------------------------- | -------------------------- | ------------------ |
  | False               | 0.6310    | Epoch 103, 0.001452, 0.7674 | 0.78466, 0.94088           | 0.78476, 0.94098   |
  | True, group_size 16 | 0.6348    | Epoch 103, 0.001388, 0.7691 | 0.78664, 0.94192           | 0.78630, 0.94192   |
***

# Progressive training
## EfficientNetV2B0 cifar10 basic test
  - Refer to [PDF 2104.00298 EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf).
  ```sh
  # Normally training input_shape 224, magnitude 15, dropout 0.4
  CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py \
  -m efficientnet.EfficientNetV2B0 --pretrained imagenet -d cifar10 --lr_decay_steps 36 -s effv2b0_cifar10_224_magnitude_15_dropout_0.4 \
  --epochs -1 \
  --input_shape 224 \
  --additional_model_kwargs '{"dropout": 0.4}' \
  --magnitude 15 \
  --batch_size 240 \
  --seed 0
  ```
  ```sh
  #  4 stages progressive training input_shape [128, 160, 192, 224]
  CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./progressive_train_script.py \
  -m efficientnet.EfficientNetV2B0 --pretrained imagenet -d cifar10 --lr_decay_steps 36 -s effv2b0_cifar10_224_progressive \
  --progressive_epochs 10 20 30 -1 \
  --progressive_input_shapes 128 160 192 224 \
  --progressive_dropouts 0.1 0.2 0.3 0.4 \
  --progressive_magnitudes 5 8 12 15 \
  --progressive_batch_sizes 240 \
  --seed 0
  ```
  ![progressive_cifar10](https://user-images.githubusercontent.com/5744524/147729276-fd9120dc-3692-4674-ad42-d197910bb588.png)
## AotNet50 A3 progressive 96 128 160
  ```sh
  CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./progressive_train_script.py \
  --progressive_epochs 33 66 -1 \
  --progressive_input_shapes 96 128 160 \
  --progressive_magnitudes 2 4 6 \
  -s aotnet50_progressive_3_lr_steps_100 --seed 0
  ```

  | progressive  | Train acc | Best eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
  | ------------ | --------- | --------------------------- | -------------------------- | ------------------ |
  | None         | 0.6310    | Epoch 103, 0.001452, 0.7674 | 0.78466, 0.94088           | 0.78476, 0.94098   |
  | 96, 128, 160 | 0.6293    | Epoch 101, 0.001438, 0.7672 | 0.78074, 0.93912           | 0.78090, 0.93912   |

  ![aotnet50_progressive_160](https://user-images.githubusercontent.com/5744524/151286851-221ff8eb-9fe9-4685-aa60-4a3ba98c654e.png)
***
