import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import functional


@backend.register_keras_serializable(package="kecamLoss")
def clip_loss(y_true, y_pred):
    caption_loss = backend.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    image_loss = backend.losses.sparse_categorical_crossentropy(y_true, functional.transpose(y_pred), from_logits=True)
    return (caption_loss + image_loss) / 2.0
