## Visualizing
***

## Summary
***
## Usage
- Displaying the visual patterns that convnet filters respond to. Copie from [keras.io/examples Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn/).
```py
>>> from keras_cv_attention_models import visualizing
>>> model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
>>> losses, all_images = visualizing.visualize_filters(model, "conv3_block4_out", [0], 180, 180)
>>> print(f"{losses[0].numpy() = }, {all_images[0].shape = }")
# losses[0].numpy() = 13.749493, all_images[0].shape = (130, 130, 3)
>>> plt.imshow(all_images[0])
```
```py
>>> from keras_cv_attention_models import visualizing
>>> model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
>>> losses, all_images = visualizing.visualize_filters(model, "conv3_block4_out", range(10), 180, 180)
>>> print(f"{losses[0].numpy() = }, {len(all_images) = }, {all_images[0].shape = }")
# losses[0].numpy() = 13.749493, len(all_images) = 10, all_images[0].shape = (130, 130, 3)
>>> image = visualizing.visualize_filters_result_to_single_image(all_images)
>>> print(f"{image.shape = }")
# image.shape = (265, 670, 3)
>>> plt.imshow(image)
```
