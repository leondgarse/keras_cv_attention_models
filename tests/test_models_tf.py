import pytest
import sys

sys.path.append(".")
import keras_cv_attention_models
from keras_cv_attention_models.backend import models
from keras_cv_attention_models.test_images import cat

""" Recognition models CotNet / HaloNet / HorNet / NFNet / VOLO defination """


def test_CotNet_defination():
    mm = keras_cv_attention_models.cotnet.CotNet50(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.cotnet.CotNetSE101D(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_HaloNet_defination():
    mm = keras_cv_attention_models.halonet.HaloNetH0(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.halonet.HaloNetH2(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_HorNet_defination():
    mm = keras_cv_attention_models.hornet.HorNetSmall(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.hornet.HorNetSmallGF(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_NFNet_defination():
    mm = keras_cv_attention_models.nfnets.NFNetF0(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.nfnets.ECA_NFNetL1(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_VOLO_defination():
    mm = keras_cv_attention_models.volo.VOLO_d3(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.volo.VOLO_d4(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


""" Recognition models EfficientNetV2B1_preprocessing / HaloNet / HorNet / VOLO prediction """


def test_EfficientNetV2B1_preprocessing_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B1(pretrained="imagenet", include_preprocessing=True)
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HaloRegNetZB_predict():
    mm = keras_cv_attention_models.halonet.HaloRegNetZB(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HorNetTinyGF_new_shape_predict():
    mm = keras_cv_attention_models.hornet.HorNetTinyGF(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_VOLO_d1_new_shape_predict():
    mm = keras_cv_attention_models.volo.VOLO_d1(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
