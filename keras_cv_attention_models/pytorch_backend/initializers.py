import torch


def zeros():
    return torch.zeros


def ones():
    return torch.ones


def truncated_normal(mean=0.0, stddev=1e-6, seed=None):
    return TruncatedNormal(mean=mean, stddev=stddev, seed=seed)


class Initializer:
    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)
        return cls(**config)


class VarianceScaling(Initializer):
    def __init__(self, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        self.scale, self.mode, self.distribution, self.seed = scale, mode, distribution, seed

    def __call__(self, shape, dtype=None, **kwargs):
        return torch.zeros(shape)  # [TODO]

    def get_config(self):
        return {"scale": self.scale, "mode": self.mode, "distribution": self.distribution}


class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        return torch.zeros(shape) + self.value

    def get_config(self):
        return {"value": self.value}


class TruncatedNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean, self.stddev, self.seed = mean, stddev, seed

    def __call__(self, shape, dtype=None, **kwargs):
        return torch.zeros(shape)  # [TODO]

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev}
