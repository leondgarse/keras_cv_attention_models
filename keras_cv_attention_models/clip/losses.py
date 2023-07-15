import math
from keras_cv_attention_models.backend import layers, models, functional

def clip_loss(y_true, y_pred) :
    # normalized features
    half_split = y_pred.shape[-1] // 2
    text_latents, image_latents = y_pred[:, :half_split], y_pred[:, half_split:]
    image_latents = image_latents / tf.norm(tensor=image_latents, ord="euclidean", axis=-1, keepdims=True)
    text_latents = text_latents / tf.norm(tensor=text_latents, ord="euclidean", axis=-1, keepdims=True)

    # cosine similarity as logits
    logits_per_text = tf.matmul(text_latents, image_latents, transpose_b=True)
    logits_per_image = tf.transpose(logits_per_text)
    similarity = logits_per_text

    caption_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(tf.range(tf.shape(similarity)[0]), similarity, from_logits=True))
    image_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(tf.range(tf.shape(similarity)[1]), tf.transpose(similarity), from_logits=True))
    return (caption_loss + image_loss) / 2.0
