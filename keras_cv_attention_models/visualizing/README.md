## Visualizing
  - Visualizing convnet filters or attention map scores.
***

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
  # >>>> Prediction: [('n02106662', 'German_shepherd', 0.83640593), ('n02123045', 'tabby', 0.016732752), ...]
  # >>>> Top 5 prediction indexes: [235 281 282 285 225]
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
    from keras_cv_attention_models import visualizing, test_images, botnet, halonet, beit, levit, coatnet
    imm = test_images.dog()
    ```
  - **BEIT** model attention score format `[batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]`.
    ```py
    _ = visualizing.plot_attention_score_maps(beit.BeitBasePatch16(), imm, rescale_mode='tf', rows=2)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209433-9dfdd736-9c92-4264-b6af-6b12d886ad36.png)
  - **LeViT** model attention score format `[batch, num_heads, q_blocks, k_blocks]`.
    ```py
    _ = visualizing.plot_attention_score_maps(levit.LeViT128S(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209475-fa4dfdbd-9a3a-4568-b139-85389cbd612e.png)
  - **BotNet** model attention score format `[batch, num_heads, hh * ww, hh * ww]`.
    ```py
    _ = visualizing.plot_attention_score_maps(botnet.BotNetSE33T(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209511-f5194d73-9e4c-457e-a763-45a4025f452b.png)
  - **HaloNet** model attention score format `[batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]`. **This one seems not right**.
    ```py
    _ = visualizing.plot_attention_score_maps(halonet.HaloNet50T(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209558-2c1c1590-20d6-4c09-9686-11521ac51b37.png)
  - **CoAtNet** model attention score format `[batch, num_heads, hh * ww, hh * ww]`. Similar with `BotNet`, but using `max_pooling`.
    ```py
    _ = visualizing.plot_attention_score_maps(coatnet.CoAtNet0(input_shape=(160, 160, 3)), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/148001256-8d123cef-0ced-491b-ae23-d59ecec418c3.png)
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
***
