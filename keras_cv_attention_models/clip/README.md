# ___Keras CLIP___
***

## Related
  - [Official, Github openai/CLIP](https://github.com/openai/CLIP)
  - [Communiy, used by timm, Github mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
  - [Paper 2103.00020 Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
  - [Paper 2111.02114 LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/abs/2111.02114)
  - [Github taki0112/CLIP-Tensorflow](https://github.com/taki0112/CLIP-Tensorflow)
  - [Github lucidrains/x-clip](https://github.com/lucidrains/x-clip)
# Requirments
  ```sh
  pip install ftfy regex sentencepiece
  ```
# Training
- Colab train test [kecam_caption_test.ipynb](https://colab.research.google.com/drive/1VaOOE4Q2rD_pV4k3YymY1glqtlNjoikT?usp=sharing).
- **Create custom dataset using custom_dataset_script.py**. Required is a `tsv` / `json` file with image path and caption info. Detail usage follow [Custom caption detaset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussioncomment-6516154). Here taking a subset from COCO caption as an example.
  ```sh
  !wget https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/coco_dog_cat.tar.gz
  !mkdir -p datasets
  !mv coco_dog_cat.tar.gz datasets && cd datasets && tar xf coco_dog_cat.tar.gz && cd -
  !head datasets/coco_dog_cat/captions.tsv
  # base_path       .
  # datasets/coco_dog_cat/train2017/images/000000014285.jpg A cat sleeping on a bed with a small TV in a bedroom.
  # datasets/coco_dog_cat/train2017/images/000000252203.jpg A couple of dogs sitting in the front seats of a car.
  # datasets/coco_dog_cat/train2017/images/000000159075.jpg A dog standing on top of a pickup truck
  ```
  Or creat dataset using entire COCO captions
  ```sh
  DATA_PATH=/datasets/coco/2017
  python custom_dataset_script.py --train_images $DATA_PATH/train2017 --test_images $DATA_PATH/val2017 \
  --train_captions $DATA_PATH/annotations/captions_train2017.json \
  --test_captions $DATA_PATH/annotations/captions_val2017.json -s coco_captions
  # >>>> total_train_samples: 591753, totla_test_samples: 25014
  # >>>> Saved to: coco_captions.tsv
  ```
- **Train using `clip_train_script.py on COCO captions`** Default `--data_path` is a testing one `datasets/coco_dog_cat/captions.tsv`.
  ```sh
  CUDA_VISIBLE_DEVICES=1 TF_XLA_FLAGS="--tf_xla_auto_jit=2" python clip_train_script.py -i 160 -b 128 \
  --text_model_pretrained None --data_path coco_captions.tsv
  ```
  **Train Using PyTorch backend by setting `KECAM_BACKEND='torch'`**
  ```sh
  KECAM_BACKEND='torch' CUDA_VISIBLE_DEVICES=1 python clip_train_script.py -i 160 -b 128 \
  --text_model_pretrained None --data_path coco_captions.tsv
  ```
  ![clip_torch_tf](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/4cbc22e4-907d-4735-81a0-41e0fc17ebc5)
- **Reload model and run prediction after training** For TF backend h5 file, model can be directly reloaded.
  ```py
  from keras_cv_attention_models import clip, test_images, plot_func
  from keras_cv_attention_models.backend import numpy_image_resize, functional

  caption_tokenizer = clip.GPT2Tokenizer()
  model = keras.models.load_model('checkpoints/clip_eva02_small_patch14_llama2_42m_tensorflow_latest.h5', compile=False)

  """ Run prediction """
  formater = "a {}"
  text_labels = [formater.format(ii) for ii in ["person", "cat", "dog", "dog and a cat"]]
  text_inputs = np.stack([caption_tokenizer(ii).astype("int64") for ii in text_labels])

  image_size = 160
  images = [test_images.cat(), test_images.dog(), test_images.dog_cat()]
  image_inputs = np.stack([numpy_image_resize(ii / 255, [image_size, image_size], method="bicubic", antialias=True) for ii in images])
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Using imagenet one
  image_inputs = ((image_inputs - mean) / std).astype("float32")
  print(f"{text_inputs.shape = }, {image_inputs.shape = }, {image_inputs.min() = }, {image_inputs.max() = }")
  # text_inputs.shape = (4, 77), image_inputs.shape = (3, 160, 160, 3), image_inputs.min() = -2.144696, image_inputs.max() = 2.70702

  similarity = functional.softmax(1 * model([image_inputs, text_inputs]), axis=-1).numpy()
  ax = plot_func.show_images_texts_similarity(images, text_labels, similarity)
  ```
  ![clip_out](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/ce2cac67-517d-43ee-bf75-8b2d90932bf0)
- **Re-build model and load weights after training** This works for both TF and Torch, and a help function `run_prediction` is added to `model`.
  ```py
  import os
  os.environ["KECAM_BACKEND"] = "torch"

  import torch, kecam

  latents_dim = 512
  caption_tokenizer = kecam.clip.GPT2Tokenizer()
  image_model = kecam.models.EVA02SmallPatch14(pretrained=None, input_shape=(3, 160, 160), num_classes=latents_dim, classifier_activation=None)
  text_model = kecam.models.LLaMA2_42M(vocab_size=caption_tokenizer.vocab_size, pretrained=None, include_top=False)
  model, image_model, text_model = kecam.clip.convert_to_clip_model(image_model, text_model, caption_tokenizer)

  ss = torch.load("checkpoints/clip_eva02_small_patch14_llama2_42m_torch_latest.pt", map_location=torch.device("cpu"))
  model.load_state_dict(ss["state_dict"])

  """ Run prediction """
  images = [kecam.test_images.cat(), kecam.test_images.dog(), kecam.test_images.dog_cat()]
  # model.run_prediction.reset(softmax_scale=100, formatter="a {}", rescale_mode="torch")
  similarity = model.run_prediction(images, ["dog and a cat", "dog", "cat", "person"])
  ax = keras.plot_func.show_images_texts_similarity(images, model.run_prediction.text_labels, similarity)
  ```
  ![clip_out_3](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/a75ee754-536c-487b-a047-c676ba025ae3)
## Single tower training
- **Specifying `--text_model image_model`** for creating text_model from image_model, using shared model blocks. **Also works for PyTorch backend**.
  ```sh
  CUDA_VISIBLE_DEVICES=1 python clip_train_script.py -m FlexiViTBase --text_model image_model \
  -d datasets/coco_dog_cat/captions.tsv -i 160 -b 64
  ```
- **Model built detail**
  - `image_model` is firstly split to 3 parts `head_model`/ `body_model` / `tail_model`, by if `block` in layer name, where `body_model` is the shared part with `text_model`.
  - Then a text input / output block is added to `body_model`.
  - Target model is built combining `image_model` and `text_model`, and outputs `similarity` result.
  ```py
  from keras_cv_attention_models import clip, beit, test_images, plot_func
  image_model = beit.FlexiViTBase(num_classes=512, classifier_activation=None)
  caption_tokenizer = clip.GPT2Tokenizer()  # Required for running prediction
  # image_model, text_model = clip.models.build_text_model_from_image_model(image_model)
  model, image_model, text_model = clip.convert_to_clip_model(image_model, caption_tokenizer=caption_tokenizer)

  print(f"{image_model.output_shape = }, {text_model.output_shape = }")
  # image_model.output_shape = (None, 512), text_model.output_shape = (None, 512)
  print(f"{model.input_shape = }, {model.output_shape = }")
  # model.input_shape = [(None, 240, 240, 3), (None, None)], model.output_shape = (None, None)
  print(f"{model([tf.ones([4, 240, 240, 3]), tf.ones([4, 77])]).shape = }")
  # model([tf.ones([4, 240, 240, 3]), tf.ones([4, 77])]).shape = TensorShape([4, 4])
  ```
