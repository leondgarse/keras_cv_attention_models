import pytest
from tensorflow import keras
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_cv_attention_models


def test_Beit_defination():
    mm = keras_cv_attention_models.beit.BeitBasePatch16(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.beit.BeitLargePatch16(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_BotNet_defination():
    mm = keras_cv_attention_models.botnet.BotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.botnet.BotNet101(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CMT_defination():
    mm = keras_cv_attention_models.cmt.CMTTiny(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.cmt.CMTXS(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CoaT_defination():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.coat.CoaTLiteSmall(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CoAtNet_defination():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.coatnet.CoAtNet2(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CotNet_defination():
    mm = keras_cv_attention_models.cotnet.CotNet50(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.cotnet.CotNetSE101D(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_CovNeXt_defination():
    mm = keras_cv_attention_models.convnext.ConvNeXtSmall(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.convnext.ConvNeXtBase(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_GMLP_defination():
    mm = keras_cv_attention_models.mlp_family.GMLPB16(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.mlp_family.GMLPS16(pretrained=None, num_classes=0)
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


def test_NFNet_defination():
    mm = keras_cv_attention_models.nfnets.NFNetF0(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.nfnets.ECA_NFNetL1(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_RegNet_defination():
    mm = keras_cv_attention_models.resnet_family.RegNetY040(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.resnet_family.RegNetZC16(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_ResMLP_defination():
    mm = keras_cv_attention_models.mlp_family.ResMLP12(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.mlp_family.ResMLP24(pretrained=None, num_classes=0)
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


def test_UniFormer_defination():
    mm = keras_cv_attention_models.uniformer.UniformerSmallPlus32(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.uniformer.UniformerSmall32(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_VOLO_defination():
    mm = keras_cv_attention_models.volo.VOLO_d3(pretrained=None)
    assert isinstance(mm, keras.models.Model)

    mm = keras_cv_attention_models.volo.VOLO_d4(pretrained=None, num_classes=0)
    assert isinstance(mm, keras.models.Model)


def test_Beit_new_shape_predict():
    mm = keras_cv_attention_models.beit.BeitBasePatch16(input_shape=(320, 320, 3))
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_BotNet_new_shape_predict():
    mm = keras_cv_attention_models.botnet.BotNextECA26T(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CoAtNet_new_shape_predict():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(input_shape=(320, 320, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_ConvNeXt_predict():
    mm = keras_cv_attention_models.convnext.ConvNeXtTiny(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HaloRegNetZB_predict():
    mm = keras_cv_attention_models.halonet.HaloRegNetZB(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_RegNetZB16_predict():
    mm = keras_cv_attention_models.resnet_family.RegNetZB16(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_new_shape_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(input_shape=(320, 320, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_UniformerSmall64_new_shape_predict():
    mm = keras_cv_attention_models.uniformer.UniformerSmall64(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_VOLO_d1_predict():
    mm = keras_cv_attention_models.volo.VOLO_d1(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_VOLO_d2_new_shape_predict():
    mm = keras_cv_attention_models.volo.VOLO_d2(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV2B0_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B0(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV2B1_preprocessing_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B1(pretrained="imagenet", include_preprocessing=True)
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV2B2_imagenet21k_ft1k_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B2(pretrained="imagenet21k-ft1k")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV1B0_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV1B0(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV1B1_noisy_student_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV1B1(pretrained="noisy_student")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientDetD0_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetD0(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 49104, 94)

    pred_label = mm.decode_predictions(pred)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDetD1_dynamic_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetD1(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (376, 227, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDetLite1_dynamic_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetLite1(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (376, 227, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDet_header():
    bb = keras_cv_attention_models.coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.efficientdet.EfficientDet(bb)

    assert mm.output_shape == (None, 12276, 94)


def test_YOLOXTiny_predict():
    mm = keras_cv_attention_models.yolox.YOLOXTiny(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 3549, 85)

    pred_label = mm.decode_predictions(pred)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOXS_dynamic_predict():
    mm = keras_cv_attention_models.yolox.YOLOXS(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 276, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 1110, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOX_header():
    bb = keras_cv_attention_models.efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.yolox.YOLOX(bb)

    assert mm.output_shape == (None, 1344, 85)


def test_YOLOR_CSP_predict():
    mm = keras_cv_attention_models.yolor.YOLOR_CSP(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 25200, 85)

    pred_label = mm.decode_predictions(pred)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOR_CSP_dynamic_predict():
    mm = keras_cv_attention_models.yolor.YOLOR_CSP(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 275, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 3330, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1]
    assert keras_cv_attention_models.coco.data.COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOR_header():
    bb = keras_cv_attention_models.efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.yolor.YOLOR(bb)

    assert mm.output_shape == (None, 4032, 85)
