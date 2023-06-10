# ___Keras Hiera___
***

## Summary
  - Keras implementation of [Github facebookresearch/hiera](https://github.com/facebookresearch/hiera). Paper [PDF 2306.00989 Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/pdf/2306.00989.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model         | Params  | FLOPs   | Input | Top1 Acc | Download |
  | ------------- | ------- | ------- | ----- | -------- | -------- |
  | HieraTiny     | 27.91M  | 4.93G   | 224   | 82.8     |          |
  | HieraSmall    | 35.01M  | 6.44G   | 224   | 83.8     |          |
  | HieraBase     | 51.52M  | 9.43G   | 224   | 84.5     |          |
  | HieraBasePlus | 69.90M  | 12.71G  | 224   | 85.2     |          |
  | HieraLarge    | 213.74M | 40.43G  | 224   | 86.1     |          |
  | HieraHuge     | 672.78M | 125.03G | 224   | 86.9     |          |
## Usage
  ```py
  from keras_cv_attention_models import hiera, test_images

  # Will download and load pretrained imagenet weights.
  mm = hiera.HieraBase()

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8966972), ('n02123045', 'tabby', 0.0072582546), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch torch_hiera """
  sys.path.append('../hiera/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from hiera import hiera as torch_hiera

  torch_model = torch_hiera.hiera_base_224()
  ss = torch.load('hiera_base_224.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model_state'])
  _ = torch_model.eval()

  """ Keras HieraBase """
  from keras_cv_attention_models import hiera
  mm = hiera.HieraBase(classifier_activation="softmax")

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=5e-2) = }")
  # np.allclose(torch_out, keras_out, atol=5e-2) = True
  ```
