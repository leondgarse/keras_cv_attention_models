import pytest
import sys
import numpy as np

sys.path.append(".")
import keras_cv_attention_models
from keras_cv_attention_models.test_images import cat


def test_RepViT_M1_use_distillation_switch_to_deploy():
    mm = keras_cv_attention_models.models.RepViT_M1(use_distillation=True, classifier_activation=None)
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose((preds[0] + preds[1]) / 2, preds_deploy, atol=1e-5)


def test_RepViT_M1_not_distillation_switch_to_deploy():
    mm = keras_cv_attention_models.models.RepViT_M1(use_distillation=False)
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-5)


def test_FasterViT0_switch_to_deploy():
    mm = keras_cv_attention_models.models.FasterViT0()
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-5)


def test_SwinTransformerV2Tiny_window8_switch_to_deploy():
    mm = keras_cv_attention_models.models.SwinTransformerV2Tiny_window8()
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-5)


def test_LeViT128S_switch_to_deploy():
    mm = keras_cv_attention_models.models.LeViT128S()
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-5)


def test_VanillaNet5_switch_to_deploy():
    mm = keras_cv_attention_models.models.VanillaNet5()
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-5)


def test_YOLO_NAS_S_switch_to_deploy():
    mm = keras_cv_attention_models.models.YOLO_NAS_S(use_reparam_conv=True)
    preds = mm(mm.preprocess_input(cat()))

    bb = mm.switch_to_deploy()
    preds_deploy = bb(bb.preprocess_input(cat()))
    assert np.allclose(preds, preds_deploy, atol=1e-3)
