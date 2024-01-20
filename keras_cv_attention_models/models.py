import os
import keras_cv_attention_models as __package__  # don't show `keras_cv_attention_models` under `keras_cv_attention_models.models.`


def register_model(model_func):
    if not hasattr(__package__.models, model_func.__name__):
        setattr(__package__.models, model_func.__name__, model_func)
    return model_func


def no_grad_if_torch(func):
    if __package__.backend.is_torch_backend:
        import torch

        def no_grad_call(*args, **kwargs):
            with torch.no_grad():
                return func(*args, **kwargs)

        return no_grad_call
    else:
        return func


class FakeModelWrapper:
    def __init__(self, models, name="model"):
        self.models = models if isinstance(models, (list, tuple)) else [models]
        self.name = name

    def cuda(self):
        """Torch function"""
        self.models = [model.cuda() for model in self.models]
        return self

    def cpu(self):
        """Torch function"""
        self.models = [model.cpu() for model in self.models]
        return self

    def float(self):
        """Torch function"""
        self.models = [model.float() for model in self.models]
        return self

    def half(self):
        """Torch function"""
        self.models = [model.half() for model in self.models]
        return self

    def to(self, *args):
        """Torch function"""
        self.models = [model.to(*args) for model in self.models]
        return self

    def _save_load_file_path_rule_(self, file_path=None):
        file_path = self.name if file_path is None else file_path
        suffix = os.path.splitext(file_path)[1]
        if suffix in [".h5", ".keras", ".pt", ".pth"]:
            file_path = os.path.splitext(file_path)[0]
            save_path_rule = lambda model_name: file_path + "_" + model_name + suffix
        else:  # Regard as directory
            if not os.path.exists(file_path):
                os.makedirs(file_path, exist_ok=True)
            save_path_rule = lambda model_name: os.path.join(file_path, model_name + ".h5")
        return save_path_rule

    def save(self, file_path=None):
        """file_path: if suffix in [".h5", ".keras", ".pt", ".pth"], will save as {file_path}_{model_name}.{suffix},
        or will regard as directory, and save to {file_path}/{model_name}.h5
        """
        save_path_rule = self._save_load_file_path_rule_(file_path)
        for model in self.models:
            cur_save_path = save_path_rule(model.name)
            print(">>>> Saving {} to {}".format(model.name, cur_save_path))
            model.save(cur_save_path)

    def save_weights(self, file_path=None):
        """file_path: if suffix in [".h5", ".keras", ".pt", ".pth"], will save as {file_path}_{model_name}.{suffix},
        or will regard as directory, and save to {file_path}/{model_name}.h5
        """
        save_path_rule = self._save_load_file_path_rule_(file_path)
        for model in self.models:
            cur_save_path = save_path_rule(model.name)
            print(">>>> Saving {} weights to {}".format(model.name, cur_save_path))
            model.save_weights(cur_save_path)

    def load_weights(self, file_path=None):
        """file_path: if suffix in [".h5", ".keras", ".pt", ".pth"], will load from {file_path}_{model_name}.{suffix},
        or will regard as directory, and load from {file_path}/{model_name}.h5
        """
        save_path_rule = self._save_load_file_path_rule_(file_path)
        for model in self.models:
            cur_save_path = save_path_rule(model.name)
            print(">>>> Loading {} from {}".format(model.name, cur_save_path))
            model.load_weights(cur_save_path)
