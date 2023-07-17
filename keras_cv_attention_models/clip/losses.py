import math
import tensorflow as tf

def clip_loss(y_true, y_pred) :
    # normalized features
    half_split = y_pred.shape[-1] // 2
    text_latents, image_latents = y_pred[:, :half_split], y_pred[:, half_split:]
    image_latents = image_latents / tf.norm(tensor=image_latents, ord="euclidean", axis=-1, keepdims=True)
    text_latents = text_latents / tf.norm(tensor=text_latents, ord="euclidean", axis=-1, keepdims=True)

    # cosine similarity as logits
    # logits_per_text = tf.matmul(text_latents, image_latents, transpose_b=True)
    # logits_per_image = tf.transpose(logits_per_text)
    similarity = tf.matmul(text_latents, image_latents, transpose_b=True)

    # y_true = tf.range(tf.shape(similarity)[0])
    caption_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, similarity, from_logits=True))
    image_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, tf.transpose(similarity), from_logits=True))
    return (caption_loss + image_loss) / 2.0
