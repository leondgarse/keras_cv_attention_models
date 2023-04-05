import pytest
from skimage.data import chelsea

import sys

sys.path.append(".")
import keras_cv_attention_models
from keras_cv_attention_models.backend import models
from keras_cv_attention_models.coco.data import COCO_80_LABEL_DICT, COCO_90_LABEL_DICT

""" Recognition models defination """


def test_Beit_defination():
    mm = keras_cv_attention_models.beit.BeitBasePatch16(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.beit.BeitLargePatch16(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_BotNet_defination():
    mm = keras_cv_attention_models.botnet.BotNet50(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.botnet.BotNet101(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_CMT_defination():
    mm = keras_cv_attention_models.cmt.CMTTiny(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.cmt.CMTTiny_torch(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_CoaT_defination():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.coat.CoaTLiteSmall(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_CoAtNet_defination():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.coatnet.CoAtNet2(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_CovNeXt_defination():
    mm = keras_cv_attention_models.convnext.ConvNeXtSmall(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.convnext.ConvNeXtBase(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_EdgeNeXt_defination():
    mm = keras_cv_attention_models.edgenext.EdgeNeXt_Small(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.edgenext.EdgeNeXt_X_Small(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_GCViT_defination():
    mm = keras_cv_attention_models.gcvit.GCViT_XXTiny(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.gcvit.GCViT_XTiny(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_GMLP_defination():
    mm = keras_cv_attention_models.mlp_family.GMLPB16(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.mlp_family.GMLPS16(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_LeViT_defination():
    mm = keras_cv_attention_models.levit.LeViT128(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.levit.LeViT192(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_MobileViT_defination():
    mm = keras_cv_attention_models.mobilevit.MobileViT_XXS(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.mobilevit.MobileViT_XS(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_MLPMixer_defination():
    mm = keras_cv_attention_models.mlp_family.MLPMixerB16(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.mlp_family.MLPMixerS16(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_RegNet_defination():
    mm = keras_cv_attention_models.resnet_family.RegNetY040(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.resnet_family.RegNetZC16(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_ResMLP_defination():
    mm = keras_cv_attention_models.mlp_family.ResMLP12(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.mlp_family.ResMLP24(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_ResNest_defination():
    mm = keras_cv_attention_models.resnest.ResNest50(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.resnest.ResNest101(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_ResNetD_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet50D(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNet101D(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_ResNetQ_defination():
    mm = keras_cv_attention_models.resnet_family.ResNet51Q(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNet61Q(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_ResNeXt_defination():
    mm = keras_cv_attention_models.resnet_family.ResNeXt50(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.resnet_family.ResNeXt101(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_SwinTransformerV2Tiny_defination():
    mm = keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2Tiny_window8(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2Tiny_ns(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_UniFormer_defination():
    mm = keras_cv_attention_models.uniformer.UniformerSmallPlus32(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.uniformer.UniformerSmall32(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


def test_WaveMLP_defination():
    mm = keras_cv_attention_models.wave_mlp.WaveMLP_T(pretrained=None)
    assert isinstance(mm, models.Model)

    mm = keras_cv_attention_models.wave_mlp.WaveMLP_S(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)


""" Recognition models prediction """


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


def test_CAFormerS18_new_shape_predict():
    mm = keras_cv_attention_models.caformer.CAFormerS18(input_shape=(112, 193, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CMTTiny_new_shape_predict():
    mm = keras_cv_attention_models.cmt.CMTTiny(input_shape=(117, 192, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CoaT_new_shape_predict():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(input_shape=(193, 117, 3), pretrained="imagenet")
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


def test_ConvNeXtV2_predict():
    mm = keras_cv_attention_models.convnext.ConvNeXtV2Atto(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_DaViT_T_new_shape_predict():
    mm = keras_cv_attention_models.davit.DaViT_T(input_shape=(376, 227, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EdgeNeXt_XX_Small_new_shape_predict():
    mm = keras_cv_attention_models.edgenext.EdgeNeXt_XX_Small(input_shape=(174, 269, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientFormerL1_new_shape_predict():
    mm = keras_cv_attention_models.efficientformer.EfficientFormerL1(input_shape=(192, 113, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientFormerV2S0_new_shape_predict():
    mm = keras_cv_attention_models.efficientformer.EfficientFormerV2S0(input_shape=(192, 113, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV1B1_noisy_student_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV1B1(pretrained="noisy_student")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV2B0_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B0(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FasterNetT2_dynamic_predict():
    mm = keras_cv_attention_models.fasternet.FasterNetT2(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FBNetV3B_dynamic_predict():
    mm = keras_cv_attention_models.fbnetv3.FBNetV3B(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GCViT_XXTiny_new_shape_predict():
    mm = keras_cv_attention_models.gcvit.GCViT_XXTiny(input_shape=(128, 192, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GhostNetV2_100_dynamic_predict():
    mm = keras_cv_attention_models.ghostnet.GhostNetV2_100(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(112, 157, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GPViT_L1_new_shape_predict():
    mm = keras_cv_attention_models.gpvit.GPViT_L1(input_shape=(112, 157, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_IFormerSmall_new_shape_predict():
    mm = keras_cv_attention_models.iformer.IFormerSmall(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_InceptionNeXtTiny_dynamic_predict():
    mm = keras_cv_attention_models.inceptionnext.InceptionNeXtTiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LCNet050_dynamic_predict():
    mm = keras_cv_attention_models.lcnet.LCNet050(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_new_shape_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(input_shape=(292, 213, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MaxViT_Tiny_new_shape_predict():
    mm = keras_cv_attention_models.maxvit.MaxViT_Tiny(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileNetV3Small075_dynamic_predict():
    mm = keras_cv_attention_models.mobilenetv3.MobileNetV3Small075(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileViT_XXS_predict():
    mm = keras_cv_attention_models.mobilevit.MobileViT_XXS(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileViT_V2_050_predict():
    mm = keras_cv_attention_models.mobilevit.MobileViT_V2_050(input_shape=(252, 224, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MogaNetXtiny_dynamic_predict():
    mm = keras_cv_attention_models.moganet.MogaNetXtiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_PVT_V2B0_new_shape_predict():
    mm = keras_cv_attention_models.pvt.PVT_V2B0(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_PVT_V2B2_linear_new_shape_predict():
    mm = keras_cv_attention_models.pvt.PVT_V2B2_linear(input_shape=(193, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_RegNetZB16_predict():
    mm = keras_cv_attention_models.resnet_family.RegNetZB16(pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_ResNest50_dynamic_predict():
    mm = keras_cv_attention_models.resnest.ResNest50(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 192, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_TinyViT_5M_new_shape_predict():
    mm = keras_cv_attention_models.tinyvit.TinyViT_5M(input_shape=(160, 160, 3))
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_TinyNetD_dynamic_predict():
    mm = keras_cv_attention_models.tinynet.TinyNetD(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=(160, 256, 3)))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_SwinTransformerV2Tiny_window8_new_shape_predict():
    mm = keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2Tiny_window8(input_shape=(160, 160, 3))
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_UniformerSmall64_new_shape_predict():
    mm = keras_cv_attention_models.uniformer.UniformerSmall64(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_WaveMLP_T_dynamic_predict():
    mm = keras_cv_attention_models.wave_mlp.WaveMLP_T(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(chelsea(), input_shape=[320, 320, 3]))  # Chelsea the cat
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


""" Detection models """


def test_EfficientDetD0_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetD0(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 49104, 94)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDet_header():
    bb = keras_cv_attention_models.coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.efficientdet.EfficientDet(bb)

    assert mm.output_shape == (None, 12276, 94)


def test_EfficientDetD1_dynamic_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetD1(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (376, 227, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDetLite1_dynamic_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetLite1(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (376, 227, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOR_CSP_predict():
    mm = keras_cv_attention_models.yolor.YOLOR_CSP(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 25200, 85)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOR_header():
    bb = keras_cv_attention_models.efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.yolor.YOLOR(bb)

    assert mm.output_shape == (None, 4032, 85)


def test_YOLOR_CSP_dynamic_predict():
    mm = keras_cv_attention_models.yolor.YOLOR_CSP(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 275, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 3330, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOXTiny_predict():
    mm = keras_cv_attention_models.yolox.YOLOXTiny(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()[:, :, ::-1]))  # Chelsea the cat
    assert pred.shape == (1, 3549, 85)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOX_header():
    bb = keras_cv_attention_models.efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
    mm = keras_cv_attention_models.yolox.YOLOX(bb)

    assert mm.output_shape == (None, 1344, 85)


def test_YOLOXS_dynamic_predict():
    mm = keras_cv_attention_models.yolox.YOLOXS(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 276, 3)
    pred = mm(mm.preprocess_input(chelsea()[:, :, ::-1], input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 1110, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV7_Tiny_predict():
    mm = keras_cv_attention_models.yolov7.YOLOV7_Tiny(pretrained="coco")
    pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
    assert pred.shape == (1, 10647, 85)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV7_Tiny_dynamic_predict():
    mm = keras_cv_attention_models.yolov7.YOLOV7_Tiny(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 276, 3)
    pred = mm(mm.preprocess_input(chelsea(), input_shape=input_shape))  # Chelsea the cat
    assert pred.shape == (1, 3330, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"
