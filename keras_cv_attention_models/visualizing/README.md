## Visualizing
***

## Visualize filters
  - Displaying the visual patterns that convnet filters respond to. Will create input images that maximize the activation of specific filters in a target layer. Such images represent a visualization of the pattern that the filter responds to. Copied and modified from [keras.io/examples Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, resnest
  mm = resnest.ResNest50()
  losses, filter_images, ax = visualizing.visualize_filters(mm, "stack3_block6_out", range(10))
  print(f"{losses[0] = }, {filter_images[0].shape = }")
  # losses[0] = 23.950336, filter_images[0].shape = (174, 174, 3)
  ```
  ![](https://user-images.githubusercontent.com/5744524/147209311-02dbb24c-6971-439f-9413-4724a34a4fc7.png)
## Make gradcam heatmap
  - Grad-CAM class activation visualization. Obtain a class activation heatmap for an image classification model. Copied and modified from: [keras.io/examples Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, resnest
  mm = resnest.ResNest50()
  url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
  img = plt.imread(keras.utils.get_file('aa.jpg', url))
  img = tf.expand_dims(tf.image.resize(img, mm.input_shape[1:-1]), 0)
  img = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
  heatmap, preds = visualizing.make_gradcam_heatmap(mm, img, 'stack4_block3_out')
  print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
  # preds.shape = (1, 1000), heatmap.shape = (7, 7), heatmap.max() = 1.0, heatmap.min() = 0.0
  print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
  # ('n02110063', 'malamute', 0.54736596)

  plt.imshow(heatmap)
  ```
  ![](https://user-images.githubusercontent.com/5744524/147209356-2a32f930-4af9-4b8f-ad2f-1b91acbb4bc3.png)
## Make and apply gradcam heatmap
  - Grad-CAM class activation visualization. Create and plot a superimposed visualization heatmap on image. Copied and modified from: [keras.io/examples Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/).
  - Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.
  ```py
  from keras_cv_attention_models import visualizing, resnest
  mm = resnest.ResNest50()
  url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
  img = plt.imread(keras.utils.get_file('aa.jpg', url))
  superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, img, "stack4_block3_out")
  ```
  ![](https://user-images.githubusercontent.com/5744524/147209399-9fe5f08f-c93e-4b0d-b1ed-f6f72f0a9a5b.png)
## Plot attention score maps
  - Visualizing model attention score maps, superimposed with specific image.
    ```py
    from keras_cv_attention_models import botnet, halonet, beit, levit, visualizing
    url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
    imm = plt.imread(keras.utils.get_file('aa.jpg', url))
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
    _ = visualizing.plot_attention_score_maps(botnet.BotNetSE33T(), imm)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209511-f5194d73-9e4c-457e-a763-45a4025f452b.png)
  - **HaloNet** model attention score format `[batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]`. **This one seems not right**.
    ```py
    _ = visualizing.plot_attention_score_maps(halonet.HaloNet50T(), imm, rescale_mode='torch')
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209558-2c1c1590-20d6-4c09-9686-11521ac51b37.png)
  - **CoAtNet** model attention score format `[batch, num_heads, hh * ww, hh * ww]`. Plot by load_model from file.
    ```py
    _ = visualizing.plot_attention_score_maps(keras.models.load_model("checkpoints/coatnet.CoAtNet0_160.h5"), imm)
    ```
    ![](https://user-images.githubusercontent.com/5744524/147209593-094a7294-7022-4a58-898e-b967570847f0.png)
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
