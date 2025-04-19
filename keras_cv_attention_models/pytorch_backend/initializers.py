import torch


""" Functions """


def _to_dtype_(tensor, dtype=None):
    if dtype is None:
        return tensor

    default_dtype = str(torch.get_default_dtype()).split(".")[-1]  # torch.float32 -> "float32"
    if dtype == default_dtype:
        return tensor
    return tensor.to(dtype=getattr(torch, dtype, torch.get_default_dtype()))


def constant(value=0):
    return Constant(value=value)


def glorot_normal():
    return GlorotNormal()


def glorot_uniform():
    return GlorotUniform()


def he_normal():
    return HeNormal()


def he_uniform():
    return HeUniform()


def ones():
    return Ones()


def random_normal(mean=0.0, stddev=1e-6, seed=None):
    return RandomNormal(mean=mean, stddev=stddev, seed=seed)


def random_uniform(minval=-0.05, maxval=0.05, seed=None):
    return RandomUniform(minval=minval, maxval=maxval, seed=seed)


def truncated_normal(mean=0.0, stddev=1e-6, seed=None):
    return TruncatedNormal(mean=mean, stddev=stddev, seed=seed)


def zeros():
    return Zeros()


""" Classes """


class Initializer:
    def __init__(self, seed=None):
        self.seed = seed

    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)
        return cls(**config)

    def get_config(self):
        return {"seed": self.seed}


class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value
        super().__init__(seed=None)

    def __call__(self, shape, dtype=None, **kwargs):
        if hasattr(self.value, "shape") and tuple(self.value.shape) == tuple(shape):
            return _to_dtype_(self.value, dtype)
        else:
            return _to_dtype_(torch.nn.init.constant_(torch.empty(shape), val=self.value), dtype)

    def get_config(self):
        return {"value": self.value}


class GlorotNormal(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.xavier_normal_(torch.empty(shape)), dtype)


class GlorotUniform(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.xavier_uniform_(torch.empty(shape)), dtype)


class HeNormal(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.kaiming_normal_(torch.empty(shape)), dtype)


class HeUniform(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.kaiming_uniform_(torch.empty(shape)), dtype)


class Ones(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.ones_(torch.empty(shape)), dtype)


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean, self.stddev = mean, stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.normal_(torch.empty(shape), mean=self.mean, std=self.stddev), dtype)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev}


class RandomUniform(Initializer):
    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval, self.maxval = minval, maxval
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.uniform_(torch.empty(shape), a=self.minval, b=self.maxval), dtype)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval}


class TruncatedNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean, self.stddev = mean, stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.trunc_normal_(torch.empty(shape), mean=self.mean, std=self.stddev), dtype)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev}


class VarianceScaling(Initializer):
    def __init__(self, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        # scale=2.0, mode="fan_in", distribution="uniform", seed=seed  # HeUniform
        # scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed  # HeNormal
        # scale=1.0, mode="fan_in", distribution="uniform", seed=seed  # LecunUniform
        # scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed  # LecunNormal
        # scale=1.0, mode="fan_avg", distribution="uniform", seed=seed  # GlorotUniform
        # scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed  # GlorotNormal
        self.scale, self.mode, self.distribution, self.seed = scale, mode, distribution, seed

    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.zeros(shape), dtype)  # [TODO]

    def get_config(self):
        return {"scale": self.scale, "mode": self.mode, "distribution": self.distribution}


class Zeros(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return _to_dtype_(torch.nn.init.zeros_(torch.empty(shape)), dtype)
