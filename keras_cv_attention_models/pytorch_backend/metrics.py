from keras_cv_attention_models.pytorch_backend import functional

BUILDIN_METRICS = {}


def register_metrics(name=None):
    def decorator(arg):
        registered_names = name or [arg.__name__]
        registered_names = registered_names if isinstance(registered_names, (list, tuple)) else [registered_names]
        for registered_name in registered_names:
            if registered_name in BUILDIN_METRICS:
                raise ValueError(f"{registered_name} has already been registered to " f"{BUILDIN_METRICS[registered_name]}")
            BUILDIN_METRICS[registered_name] = arg
        return arg

    return decorator


class Metric:
    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name = name
        self.eval_only = False
        self.reset_state()

    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass


@register_metrics(name=["acc", "accuracy"])
class Accuracy(Metric):
    def __init__(self, name="acc"):
        super().__init__(name=name)

    def reset_state(self):
        self.sum_value, self.passed_steps = 0.0, 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = functional.argmax(y_pred, axis=-1)
        if len(y_true.shape) > len(y_pred.shape):
            y_true = functional.argmax(y_true, axis=-1)
        cur_acc = functional.reduce_mean(functional.cast(y_true == y_pred, "float32"))
        self.sum_value = self.sum_value + cur_acc
        self.passed_steps += 1

    def result(self):
        return self.sum_value / self.passed_steps


@register_metrics(name=["acc5", "accuracy5"])
class Accuracy5(Metric):
    def __init__(self, name="acc5"):
        super().__init__(name=name)
        self.eval_only = True

    def reset_state(self):
        self.sum_value, self.passed_steps = 0.0, 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = functional.argsort(y_pred, direction="DESCENDING", axis=-1)[:, :5]
        if len(y_true.shape) >= len(y_pred.shape):
            y_true = functional.argmax(y_true, axis=-1)
        cur_acc = functional.reduce_mean(functional.convert_to_tensor([y_true[id] in y_pred[id] for id in range(y_true.shape[0])], "float32"))
        self.sum_value = self.sum_value + cur_acc
        self.passed_steps += 1

    def result(self):
        return self.sum_value / self.passed_steps
