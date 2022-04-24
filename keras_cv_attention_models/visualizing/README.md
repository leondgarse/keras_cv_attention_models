# Visualizing
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Summary](#summary)
- [Visualize filters](#visualize-filters)
- [Make gradcam heatmap](#make-gradcam-heatmap)
- [Make and apply gradcam heatmap](#make-and-apply-gradcam-heatmap)
- [Plot attention score maps](#plot-attention-score-maps)
- [TensorBoard Parallel Coordinates Plot](#tensorboard-parallel-coordinates-plot)

<!-- /TOC -->
***
## Summary
  - Visualizing convnet filters or attention map scores, and other display related.

## Visualize filters
  - Displaying the visual patterns that convnet filters respond to. Will create input images that maximize the activation of specific filters in a target layer. Such images represent a visualization of the pattern that the filter responds to. Copied and modified from [keras.io/examples Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, resnest
  mm = resnest.ResNest50()
  losses, filter_images, ax = visualizing.visualize_filters(mm, "stack3_block6_output", filter_index_list=range(10))
  print(f"{losses[0] = }, {filter_images[0].shape = }")
  # losses[0] = 9.274959, filter_images[0].shape = (180, 180, 3)
  ```
  ![](https://user-images.githubusercontent.com/5744524/148632851-ec1fe51c-4c4b-4cc5-ab50-2b03db981e08.png)
## Make gradcam heatmap
  - Grad-CAM class activation visualization. Obtain a class activation heatmap for an image classification model. Copied and modified from: [keras.io/examples Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, test_images, resnest
  mm = resnest.ResNest50()
  img = test_images.dog()
  img = tf.expand_dims(tf.image.resize(img, mm.input_shape[1:-1]), 0)
  img = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
  heatmap, preds = visualizing.make_gradcam_heatmap(mm, img, layer_name="auto")
  print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
  # preds.shape = (1, 1000), heatmap.shape = (7, 7), heatmap.max() = 1.0, heatmap.min() = 0.0
  print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
  # ('n02110063', 'malamute', 0.54736596)

  plt.imshow(heatmap)
  ```
  ![](https://user-images.githubusercontent.com/5744524/148199420-539f259e-f845-488f-87e1-366ff93c65dc.png)
## Make and apply gradcam heatmap
  - Grad-CAM class activation visualization. Create and plot a superimposed visualization heatmap on image. Copied and modified from: [keras.io/examples Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, test_images, resnest
  mm = resnest.ResNest50()
  img = test_images.dog_cat()
  superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, img, layer_name="auto")
  # >>>> Top5 predictions: [['235' 'n02106662' 'German_shepherd' '0.7492399']
  #  ['281' 'n02123045' 'tabby' '0.033892266']
  #  ['285' 'n02124075' 'Egyptian_cat' '0.017182153']
  #  ['282' 'n02123159' 'tiger_cat' '0.015299492']
  #  ['225' 'n02105162' 'malinois' '0.012337279']]
  ```
  ![](https://user-images.githubusercontent.com/5744524/148197367-73cb1f9b-3edf-4e95-83e1-4a77d1f4a9fd.png)
  ```py
  # Plot cat heatmap
  _ = visualizing.make_and_apply_gradcam_heatmap(mm, img, layer_name="auto", pred_index=281)
  ```
  ![](https://user-images.githubusercontent.com/5744524/148199220-8120cd4c-132a-4ca4-8694-e6c550edbb13.png)
## Plot attention score maps
  - Visualizing model attention score maps, superimposed with specific image.
    ```py
    from keras_cv_attention_models import visualizing, test_images, botnet, halonet, beit, levit, coatnet, coat
    imm = test_images.dog()
    ```
  - **beit** `attn_type`, attention score format `[batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]`.
    ```py
    _ = visualizing.plot_attention_score_maps(beit.BeitBasePatch16(), imm, rescale_mode='tf', rows=2)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209433-9dfdd736-9c92-4264-b6af-6b12d886ad36.png)
  - **levit** `attn_type`, attention score format `[batch, num_heads, q_blocks, k_blocks]`.
    ```py
    _ = visualizing.plot_attention_score_maps(levit.LeViT128S(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209475-fa4dfdbd-9a3a-4568-b139-85389cbd612e.png)
  - **bot** `attn_type`, attention score format `[batch, num_heads, hh * ww, hh * ww]`.
    ```py
    _ = visualizing.plot_attention_score_maps(botnet.BotNetSE33T(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209511-f5194d73-9e4c-457e-a763-45a4025f452b.png)
  - **halo** `attn_type`, attention score format `[batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]`. **This one seems not right**.
    ```py
    _ = visualizing.plot_attention_score_maps(halonet.HaloNet50T(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209558-2c1c1590-20d6-4c09-9686-11521ac51b37.png)
  - **coatnet / cmt / uniformer / swin** `attn_type`, attention score format `[batch, num_heads, hh * ww, hh * ww]`. Similar with `BotNet`, but using `max_pooling`.
    ```py
    _ = visualizing.plot_attention_score_maps(coatnet.CoAtNet0(input_shape=(160, 160, 3)), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/148001256-8d123cef-0ced-491b-ae23-d59ecec418c3.png)
  - **coat** `attn_type`, attention score format `[batch, num_heads, cls_token + hh * ww, key_dim]`.
    ```py
    _ = visualizing.plot_attention_score_maps(coat.CoaTTiny(), imm)
    ```
    ![](https://user-images.githubusercontent.com/5744524/164968729-0b1a89ef-67ef-4202-89e3-10c383e87379.png)
  - **VIT** model attention score format is same with `BEIT`. Plot by extract attention scores and specify attn_type.
    ```py
    from vit_keras import vit, layers
    mm = vit.vit_b16(image_size=384, activation='sigmoid', pretrained=True, include_top=True, pretrained_top=True)
    img = vit.preprocess_inputs(tf.image.resize(imm, mm.input_shape[1:-1]))[np.newaxis, :]
    outputs = [ii.output[1] for ii in mm.layers if isinstance(ii, layers.TransformerBlock)]
    attn_scores = np.array(tf.keras.models.Model(inputs=mm.inputs, outputs=outputs).predict(img))
    _ = visualizing.plot_attention_score_maps(attn_scores, imm, attn_type='beit', rows=2)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209624-5e10e7e2-2120-48cb-bc13-6761c5348a32.png)
## TensorBoard Parallel Coordinates Plot
  - Simmilar results with [Visualize the results in TensorBoard's HParams plugin](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams#4_visualize_the_results_in_tensorboards_hparams_plugin). Wrapped function just plotting ignoring training in the tutorial. The logic is using `metrics_name` specified key as metrics, using other columns as `HParams`. For any other detail, refer original tutorial.
  ```py
  import pandas as pd
  aotnet50_imagnet_results = {
    "optimizer": ["lamb", "lamb", "adamw", "adamw", "adamw"],
    "rescale_mode": ["torch", "tf", "torch", "torch", "torch"],
    "lr_base": [8e-3, 8e-3, 4e-3, 4e-3, 8e-3],
    "weight_decay": [0.05, 0.05, 0.05, 0.02, 0.02],
    "accuracy": [78.48, 78.31, 77.92, 78.06, 78.27],
  }
  aa = pd.DataFrame(aotnet50_imagnet_results)

  from keras_cv_attention_models import visualizing
  visualizing.tensorboard_parallel_coordinates_plot(aa, 'accuracy', log_dir="logs/aotnet50_imagnet_results")
  # >>>> Start tensorboard by: ! tensorboard --logdir logs/aotnet50_imagnet_results
  # >>>> Then select `HPARAMS` -> `PARALLEL COORDINATES VIEW`

  ! tensorboard --logdir logs/aotnet50_imagnet_results
  # TensorBoard 2.8.0 at http://localhost:6006/ (Press CTRL+C to quit)
  ```
  ![aotnet50_imagenet_parallel_coordinates](https://user-images.githubusercontent.com/5744524/164968989-7a443fe8-48e2-486a-995a-fe469e171088.png)
***
