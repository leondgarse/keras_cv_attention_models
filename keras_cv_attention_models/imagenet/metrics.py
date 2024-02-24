from keras_cv_attention_models.backend import metrics


class LossMeanMetricWrapper(metrics.Metric):
    def __init__(self, loss_func, loss_attr_name):
        self.loss_func, self.loss_attr_name = loss_func, loss_attr_name
        super().__init__(name=loss_attr_name)

    def reset_state(self):
        self.value, self.passed_steps = 0.0, 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value += getattr(self.loss_func, self.loss_attr_name)
        self.passed_steps += 1

    def result(self):
        return self.value / self.passed_steps
