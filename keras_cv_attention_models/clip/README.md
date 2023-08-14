# ___[Experimental] CLIP___
***

# Requirments
  ```sh
  pip install ftfy regex
  ```
# Training
- Colab train test [kecam_caption_test.ipynb](https://colab.research.google.com/drive/1VaOOE4Q2rD_pV4k3YymY1glqtlNjoikT?usp=sharing).
- **Create custom dataset using custom_dataset_script.py**. Required is a `tsv` / `json` file with image path and caption info. Detail usage follow [Custom caption detaset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussioncomment-6516154). Here taking a subset from COCO caption as an example.
  ```sh
  !wget https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/coco_dog_cat.tar.gz
  !mkdir -p datasets
  !mv coco_dog_cat.tar.gz datasets && cd datasets && tar xf coco_dog_cat.tar.gz && cd -
  !head datasets/coco_dog_cat/captions.json
  # {
  #   "train": [
  #     {"image": "train2017/images/000000000042.jpg", "caption": "This wire metal rack holds several pairs of shoes and sandals"},
  #     {"image": "train2017/images/000000000042.jpg", "caption": "A dog sleeping on a show rack in the shoes."},
  ```
- **Train using `train_script.py`** by specifying `--text_model` a text model and `--data_name` a caption dataset.
  ```sh
  CUDA_VISIBLE_DEVICES=1 python clip_train_script.py -m EVA02SmallPatch14 --text_model LLaMA2_42M \
  -d datasets/coco_dog_cat/captions.tsv -i 160 -b 128 --text_model_pretrained None
  ```
- **Reload model and run prediction after training**
  ```py
  from keras_cv_attention_models import clip, test_images

  caption_tokenizer = clip.GPT2Tokenizer()
  model = keras.models.load_model('checkpoints/clip_eva02_small_patch14_llama2_42m_tensorflow_latest.h5', compile=False)
  image_model, text_model = clip.split_to_image_text_model(model)
  run_prediction = clip.RunPrediction(image_model, text_model, caption_tokenizer)

  """ Run prediction """
  images = [test_images.cat(), test_images.dog(), test_images.dog_cat()]
  similarity = run_prediction(images, ['cat', 'dog', 'person', 'computer'])
  ax = plot_func.show_images_texts_similarity(images, run_prediction.text_labels, similarity)
  ```
- **Train Using PyTorch backend by setting `KECAM_BACKEND='torch'`** Note: saved `h5` is weights only, not supporting `keras.models.load_model`
  ```sh
  KECAM_BACKEND='torch' CUDA_VISIBLE_DEVICES=1 python clip_train_script.py -m EVA02SmallPatch14 --text_model LLaMA2_42M \
  -d datasets/coco_dog_cat/captions.tsv -i 160 -b 128 --text_model_pretrained None
  ```
## Single tower training
- **Specifying `--text_model image_model`** for creating text_model from image_model, using shared model blocks.
  ```sh
  CUDA_VISIBLE_DEVICES=1 python clip_train_script.py -m FlexiViTBase --text_model image_model \
  -d datasets/coco_dog_cat/captions.tsv -i 160 -b 128
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
  ```py
  """ Run prediction """
  images = [test_images.cat(), test_images.dog(), test_images.dog_cat()]
  similarity = model.run_prediction(images, ['cat', 'dog', 'person', 'computer'])
  ax = plot_func.show_images_texts_similarity(images, model.run_prediction.text_labels, similarity)
  ```
