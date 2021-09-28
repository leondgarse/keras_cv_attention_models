import pytest
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_cv_attention_models


def test_BotNet50_defination():
    mm = keras_cv_attention_models.botnet.BotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_CoaTLiteMini_defination():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_CoAtNet0_defination():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_CotNet50_defination():
    mm = keras_cv_attention_models.cotnet.CotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_SECotNetD50_defination():
    mm = keras_cv_attention_models.cotnet.SECotNetD50(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_HaloNetH0_defination():
    mm = keras_cv_attention_models.halonet.HaloNetH0(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_LeViT128_defination():
    mm = keras_cv_attention_models.levit.LeViT128(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_MLPMixerB16_defination():
    mm = keras_cv_attention_models.mlp_family.MLPMixerB16(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ResMLP12_defination():
    mm = keras_cv_attention_models.mlp_family.ResMLP12(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_GMLPB16_defination():
    mm = keras_cv_attention_models.mlp_family.GMLPB16(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ResNest50_defination():
    mm = keras_cv_attention_models.resnest.ResNest50(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ResNet50D_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet50D(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ResNet51Q_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet51Q(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ResNeXt50_defination():
    mm = keras_cv_attention_models.resnet_family.ResNeXt50(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_VOLO_d2_defination():
    mm = keras_cv_attention_models.volo.VOLO_d2(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_NFNetF0_defination():
    mm = keras_cv_attention_models.nfnets.NFNetF0(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_ECA_NFNetL0_defination():
    mm = keras_cv_attention_models.nfnets.ECA_NFNetL0(pretrained=None)
    assert isinstance(mm, keras.models.Model)


def test_LeViT128S_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), mm.input_shape[1:3])  # Chelsea the cat
    pred = mm(tf.expand_dims(imm / 128 - 1, 0))
    pred = ((pred[0] + pred[1]) / 2).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_VOLO_d1_predict():
    mm = keras_cv_attention_models.volo.VOLO_d1(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), mm.input_shape[1:3])  # Chelsea the cat
    pred = mm(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
