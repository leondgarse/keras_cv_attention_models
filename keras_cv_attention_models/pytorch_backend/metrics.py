from keras_cv_attention_models.pytorch_backend import functional

BUILDIN_METRICS = {}


def register_metrics(name=None):
    def decorator(arg):
        registered_name = name or arg.__name__
        if registered_name in BUILDIN_METRICS:
            raise ValueError(f"{registered_name} has already been registered to " f"{BUILDIN_METRICS[registered_name]}")
        BUILDIN_METRICS[registered_name] = arg
        return arg

    return decorator


class Metric:
    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name = name
        self.reset_state()

    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass


@register_metrics(name="acc")
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
