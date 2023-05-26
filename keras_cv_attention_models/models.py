import keras_cv_attention_models as __package__


def register_model(model_func):
    if not hasattr(__package__.models, model_func.__name__):
        setattr(__package__.models, model_func.__name__, model_func)
    return model_func
