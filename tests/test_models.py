import pytest
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_cv_attention_models


def test_BotNet_defination():
    mm = keras_cv_attention_models.botnet.BotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.botnet.BotNet101(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CoaT_defination():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.coat.CoaTLiteSmall(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CMT_defination():
    mm = keras_cv_attention_models.cmt.CMTTiny(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.cmt.CMTXS(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CoAtNet_defination():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.coatnet.CoAtNet2(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CotNet_defination():
    mm = keras_cv_attention_models.cotnet.CotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.cotnet.CotNet101(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_SECotNetD_defination():
    mm = keras_cv_attention_models.cotnet.CotNetSE50D(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.cotnet.CotNetSE101D(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_HaloNet_defination():
    mm = keras_cv_attention_models.halonet.HaloNetH0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.halonet.HaloNetH2(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_LeViT_defination():
    mm = keras_cv_attention_models.levit.LeViT128(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.levit.LeViT192(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_MLPMixer_defination():
    mm = keras_cv_attention_models.mlp_family.MLPMixerB16(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.mlp_family.MLPMixerS16(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResMLP_defination():
    mm = keras_cv_attention_models.mlp_family.ResMLP12(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.mlp_family.ResMLP24(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_GMLP_defination():
    mm = keras_cv_attention_models.mlp_family.GMLPB16(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.mlp_family.GMLPS16(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_NFNetF_defination():
    mm = keras_cv_attention_models.nfnets.NFNetF0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.nfnets.NFNetF2(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ECA_NFNetL_defination():
    mm = keras_cv_attention_models.nfnets.ECA_NFNetL0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.nfnets.ECA_NFNetL1(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResNest_defination():
    mm = keras_cv_attention_models.resnest.ResNest50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.resnest.ResNest101(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResNetD_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet50D(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNet101D(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResNetQ_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet51Q(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNet61Q(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResNeXt_defination():
    mm = keras_cv_attention_models.resnet_family.ResNeXt50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNeXt101(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_VOLO_defination():
    mm = keras_cv_attention_models.volo.VOLO_d3(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.volo.VOLO_d4(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_HaloRegNetZB_predict():
    mm = keras_cv_attention_models.halonet.HaloRegNetZB(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), mm.input_shape[1:3])  # Chelsea the cat
    pred = mm(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_RegNetZB_predict():
    mm = keras_cv_attention_models.resnet_family.RegNetZB(pretrained="imagenet")
    imm = tf.image.resize(chelsea(), mm.input_shape[1:3])  # Chelsea the cat
    pred = mm(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


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


def test_VOLO_d2_predict():
    mm = keras_cv_attention_models.volo.VOLO_d2(input_shape=(512, 512, 3), pretrained="imagenet")
    imm = tf.image.resize(chelsea(), mm.input_shape[1:3])  # Chelsea the cat
    pred = mm(tf.expand_dims(imm / 128 - 1, 0)).numpy()
    out = keras.applications.imagenet_utils.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"
