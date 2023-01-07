import numpy as np
import tensorflow as tf


class ModelWithGradForTFLite(tf.Module):
    """From [On-Device Training with TensorFlow Lite](https://www.tensorflow.org/lite/examples/on_device_training/overview?hl=ko#tensorflow_model_for_training)
    Wrapper a model with `gradcam` / `train` signature, so these process can be execute after converting TFLite.

    Example:
    >>> from keras_cv_attention_models import efficientnet, test_images
    >>> mm = efficientnet.EfficientNetV2B0()

    >>> from keras_cv_attention_models.visualizing.gradcam_heatmap import ModelWithGradForTFLite, grads_to_heatmap, apply_heatmap
    >>> saved_model = mm.name
    >>> bb = ModelWithGradForTFLite(mm)
    >>> signatures = {'serving_default': bb.serving_default.get_concrete_function(), 'gradcam': bb.gradcam.get_concrete_function()}
    >>> tf.saved_model.save(bb, saved_model, signatures=signatures)
    >>> # Convert the model
    >>> converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    >>> # enable TensorFlow and TensorFlow ops.
    >>> converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    >>> tflite_model = converter.convert()
    >>> open(mm.name + ".tflite", "wb").write(tflite_model)  # Will save a much larger model than without these signatures
    >>>
    >>> interpreter = tf.lite.Interpreter(model_content=tflite_model)
    >>> interpreter.allocate_tensors()
    >>> gradcam = interpreter.get_signature_runner("gradcam")
    >>>
    >>> image = test_images.dog_cat()
    >>> gg = gradcam(inputs=tf.expand_dims(tf.image.resize(image , [224, 224]) / 255, 0))
    >>> grads, last_conv_layer_output, class_channel = gg['grads'], gg['last_conv_layer_output'], gg['class_channel']
    >>> # Show
    >>> heatmap = grads_to_heatmap(grads, last_conv_layer_output, class_channel=class_channel, use_v2=True)
    >>> superimposed_img = apply_heatmap(image, heatmap, alpha=0.8, plot=True)
    """

    def __init__(self, model, layer_name="auto"):
        self.model = model
        self.input_shape = model.input_shape
        self.output_shape = model.output_shape

        if layer_name == "auto":
            for ii in model.layers[::-1]:
                if len(ii.output_shape) == 4:
                    # if isinstance(ii, tf.keras.layers.Conv2D):
                    layer_name = ii.name
                    print("Using layer_name:", layer_name)
                    break
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(model.inputs[0], [model.get_layer(layer_name).output, *model.outputs])
        self.gradcam = tf.function(func=self.__gradcam__, input_signature=[tf.TensorSpec(self.input_shape, tf.float32)])
        self.serving_default = tf.function(func=self.__serving_default__, input_signature=[tf.TensorSpec(self.input_shape, tf.float32)])

        if model.compiled_loss is not None:
            train_input_signature = [tf.TensorSpec(self.input_shape, tf.float32), tf.TensorSpec(self.output_shape, tf.float32)]
            self.train = tf.function(func=self.__train__, input_signature=train_input_signature)

    def __gradcam__(self, inputs):
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(inputs)
            pred_index = tf.argmax(preds, -1)
            class_channel = tf.gather(preds, pred_index, axis=-1)
        grads = tape.gradient(class_channel, last_conv_layer_output)
        return {"grads": grads, "last_conv_layer_output": last_conv_layer_output, "class_channel": class_channel}

    def __serving_default__(self, inputs):
        return {"predictions": self.model(inputs)}

    def __train__(self, inputs, labels):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.model.loss(labels, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}


def grads_to_heatmap(grads, last_conv_layer_output, class_channel=None, use_v2=True):
    # Separate out for TFLite model
    if use_v2:
        # gradcam_plus_plus from: https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/gradcam_plus_plus.py
        score_values = tf.reduce_sum(tf.math.exp(class_channel))
        first_derivative = score_values * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        reduction_axis = list(range(1, len(last_conv_layer_output.shape) - 1))
        global_sum = tf.reduce_sum(last_conv_layer_output, axis=reduction_axis, keepdims=True)
        alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
        alpha_denom = tf.where(second_derivative == 0.0, tf.ones_like(alpha_denom), alpha_denom)
        alphas = second_derivative / alpha_denom

        alpha_norm_constant = tf.reduce_sum(alphas, axis=reduction_axis, keepdims=True)
        alpha_norm_constant = tf.where(alpha_norm_constant == 0.0, tf.ones_like(alpha_norm_constant), alpha_norm_constant)
        alphas = alphas / alpha_norm_constant

        deep_linearization_weights = first_derivative * alphas
        deep_linearization_weights = tf.reduce_sum(deep_linearization_weights, axis=reduction_axis)
    else:
        # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
        deep_linearization_weights = tf.reduce_mean(grads, axis=list(range(0, len(grads.shape) - 1)))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    heatmap = last_conv_layer_output @ deep_linearization_weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    if len(heatmap.shape) > 2:
        heatmap = tf.reduce_mean(heatmap, list(range(0, len(heatmap.shape) - 2)))

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy().astype("float32")


def make_gradcam_heatmap(model, processed_image, layer_name="auto", pred_index=None, use_v2=True):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    if layer_name == "auto":
        for ii in model.layers[::-1]:
            if len(ii.output_shape) == 4:
                # if isinstance(ii, tf.keras.layers.Conv2D):
                layer_name = ii.name
                print("Using layer_name:", layer_name)
                break
    grad_model = tf.keras.models.Model(model.inputs[0], [model.get_layer(layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(processed_image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    grads = tf.cast(grads, preds.dtype)
    last_conv_layer_output = tf.cast(last_conv_layer_output, preds.dtype)
    return grads_to_heatmap(grads, last_conv_layer_output, class_channel=class_channel, use_v2=use_v2), preds.numpy()


def apply_heatmap(image, heatmap, alpha=0.8, plot=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # Use jet colormap to colorize heatmap. Use RGB values of the colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(tf.range(256))[:, :3]
    jet_heatmap = jet_colors[tf.cast(heatmap * 255, "uint8").numpy()]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.image.resize(jet_heatmap, (image.shape[:2]))  # [0, 1]

    # Superimpose the heatmap on original image
    image = image.astype("float32") / 255
    superimposed_img = (jet_heatmap * alpha + image).numpy()
    superimposed_img /= superimposed_img.max()

    if plot:
        fig = plt.figure()
        plt.imshow(superimposed_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return superimposed_img


def make_and_apply_gradcam_heatmap(model, image, layer_name="auto", rescale_mode="auto", pred_index=None, alpha=0.8, use_v2=True, plot=True):
    from keras_cv_attention_models.imagenet.data import init_mean_std_by_rescale_mode

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
    print(">>>> rescale_mode:", rescale_mode)

    image = np.array(image)
    image = image * 255 if image.max() < 2 else image  # Makse sure it's [0, 255]
    processed_image = tf.expand_dims(tf.image.resize(image, model.input_shape[1:-1]), 0)
    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    processed_image = (processed_image - mean) / std
    # processed_image = tf.keras.applications.imagenet_utils.preprocess_input(processed_image, mode=rescale_mode)
    heatmap, preds = make_gradcam_heatmap(model, processed_image, layer_name, pred_index=pred_index, use_v2=use_v2)

    top_5_idxes = np.argsort(preds[0])[-5:][::-1]
    if model.output_shape[-1] == 1000:
        decode_pred = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=5)[0]
        print(">>>> Top5 predictions:", np.array([[ii, *jj] for ii, jj in zip(top_5_idxes, decode_pred)]))
    else:
        print(">>>> Top5 predictions:", top_5_idxes)

    superimposed_img = apply_heatmap(image, heatmap, alpha=alpha, plot=plot)
    return superimposed_img, heatmap, preds
