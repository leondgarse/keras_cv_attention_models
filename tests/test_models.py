import pytest
import sys
import numpy as np

sys.path.append(".")
import keras_cv_attention_models
from keras_cv_attention_models.backend import models
from keras_cv_attention_models.coco.info import COCO_80_LABEL_DICT, COCO_90_LABEL_DICT
from keras_cv_attention_models.test_images import cat

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
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_BotNet_new_shape_predict():
    mm = keras_cv_attention_models.botnet.BotNextECA26T(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CAFormerS18_new_shape_predict():
    mm = keras_cv_attention_models.caformer.CAFormerS18(input_shape=(112, 193, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CMTTiny_new_shape_predict():
    mm = keras_cv_attention_models.cmt.CMTTiny(input_shape=(117, 192, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CoaT_new_shape_predict():
    mm = keras_cv_attention_models.coat.CoaTLiteMini(input_shape=(193, 117, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CoAtNet_new_shape_predict():
    mm = keras_cv_attention_models.coatnet.CoAtNet0(input_shape=(320, 320, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CotNet_new_shape_predict():
    mm = keras_cv_attention_models.cotnet.CotNet50(input_shape=(193, 117, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_ConvNeXt_predict():
    mm = keras_cv_attention_models.convnext.ConvNeXtTiny(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_ConvNeXtV2_predict():
    mm = keras_cv_attention_models.convnext.ConvNeXtV2Atto(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_CSPNeXtTiny_dynamic_predict():
    mm = keras_cv_attention_models.cspnext.CSPNeXtTiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_DaViT_T_new_shape_predict():
    mm = keras_cv_attention_models.davit.DaViT_T(input_shape=(376, 227, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_DINOv2_ViT_Small14_new_shape_predict():
    mm = keras_cv_attention_models.dinov2.DINOv2_ViT_Small14(input_shape=(224, 224, 3), patch_size=32)
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EVA02TinyPatch14_new_shape_predict():
    mm = keras_cv_attention_models.eva02.EVA02TinyPatch14(input_shape=(224, 224, 3), patch_size=16)
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EdgeNeXt_XX_Small_new_shape_predict():
    mm = keras_cv_attention_models.edgenext.EdgeNeXt_XX_Small(input_shape=(174, 269, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientFormerL1_new_shape_predict():
    mm = keras_cv_attention_models.efficientformer.EfficientFormerL1(input_shape=(192, 113, 3), use_distillation=True, pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientFormerV2S0_new_shape_predict():
    mm = keras_cv_attention_models.efficientformer.EfficientFormerV2S0(input_shape=(192, 113, 3), use_distillation=True, pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV1B1_noisy_student_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV1B1(pretrained="noisy_student")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientNetV2B0_predict():
    mm = keras_cv_attention_models.efficientnet.EfficientNetV2B0(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientViT_B1_new_shape_predict():
    mm = keras_cv_attention_models.efficientvit.EfficientViT_B1(input_shape=(192, 127, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_EfficientViT_M0_new_shape_predict():
    mm = keras_cv_attention_models.efficientvit.EfficientViT_M0(input_shape=(192, 127, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FasterNetT2_dynamic_predict():
    mm = keras_cv_attention_models.fasternet.FasterNetT2(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FasterViT0_new_shape_predict():
    mm = keras_cv_attention_models.fastervit.FasterViT0(input_shape=(192, 127, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FastViT_T8_dynamic_predict():
    mm = keras_cv_attention_models.fastvit.FastViT_T8(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(192, 127, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FastViT_SA12_new_shape_predict():
    mm = keras_cv_attention_models.fastvit.FastViT_SA12(input_shape=(192, 127, 3), pretrained="distill")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_FBNetV3B_dynamic_predict():
    mm = keras_cv_attention_models.fbnetv3.FBNetV3B(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GCViT_XXTiny_new_shape_predict():
    mm = keras_cv_attention_models.gcvit.GCViT_XXTiny(input_shape=(128, 192, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GhostNetV2_100_dynamic_predict():
    mm = keras_cv_attention_models.ghostnet.GhostNetV2_100(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(112, 157, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_GPViT_L1_new_shape_predict():
    mm = keras_cv_attention_models.gpvit.GPViT_L1(input_shape=(112, 157, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HaloRegNetZB_new_shape_predict():
    mm = keras_cv_attention_models.halonet.HaloRegNetZB(input_shape=(193, 117, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HieraTiny_predict():
    mm = keras_cv_attention_models.hiera.HieraTiny()
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_HorNetTiny_dynamic_predict():
    mm = keras_cv_attention_models.hornet.HorNetTiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_IFormerSmall_new_shape_predict():
    mm = keras_cv_attention_models.iformer.IFormerSmall(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_InceptionNeXtTiny_dynamic_predict():
    mm = keras_cv_attention_models.inceptionnext.InceptionNeXtTiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LCNet050_dynamic_predict():
    mm = keras_cv_attention_models.lcnet.LCNet050(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(use_distillation=True, pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_LeViT128S_new_shape_predict():
    mm = keras_cv_attention_models.levit.LeViT128S(use_distillation=True, input_shape=(292, 213, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    pred = (pred[0] + pred[1]) / 2
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MaxViT_Tiny_new_shape_predict():
    mm = keras_cv_attention_models.maxvit.MaxViT_Tiny(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileNetV3Small075_dynamic_predict():
    mm = keras_cv_attention_models.mobilenetv3.MobileNetV3Small075(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileViT_XXS_predict():
    mm = keras_cv_attention_models.mobilevit.MobileViT_XXS(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MobileViT_V2_050_predict():
    mm = keras_cv_attention_models.mobilevit.MobileViT_V2_050(input_shape=(252, 224, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_MogaNetXtiny_dynamic_predict():
    mm = keras_cv_attention_models.moganet.MogaNetXtiny(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_NAT_Mini_new_shape_predict():
    mm = keras_cv_attention_models.nat.NAT_Mini(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_DiNAT_Mini_new_shape_predict():
    mm = keras_cv_attention_models.nat.DiNAT_Mini(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_PVT_V2B0_new_shape_predict():
    mm = keras_cv_attention_models.pvt.PVT_V2B0(input_shape=(174, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_PVT_V2B2_linear_new_shape_predict():
    mm = keras_cv_attention_models.pvt.PVT_V2B2_linear(input_shape=(193, 255, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_RegNetZB16_predict():
    mm = keras_cv_attention_models.resnet_family.RegNetZB16(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_RepViT_M1_dynamic_predict():
    mm = keras_cv_attention_models.repvit.RepViT_M09(input_shape=(None, None, 3), use_distillation=False, pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 192, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_ResNest50_dynamic_predict():
    mm = keras_cv_attention_models.resnest.ResNest50(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 192, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_TinyViT_5M_new_shape_predict():
    mm = keras_cv_attention_models.tinyvit.TinyViT_5M(input_shape=(160, 160, 3))
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_TinyNetD_dynamic_predict():
    mm = keras_cv_attention_models.tinynet.TinyNetD(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_SwinTransformerV2Tiny_window8_new_shape_predict():
    mm = keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2Tiny_window8(input_shape=(160, 160, 3))
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_UniformerSmall64_new_shape_predict():
    mm = keras_cv_attention_models.uniformer.UniformerSmall64(input_shape=(512, 512, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_WaveMLP_T_dynamic_predict():
    mm = keras_cv_attention_models.wave_mlp.WaveMLP_T(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=[320, 320, 3]))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


def test_VanillaNet5_dynamic_predict():
    mm = keras_cv_attention_models.vanillanet.VanillaNet5(input_shape=(None, None, 3), pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat(), input_shape=(160, 256, 3)))
    out = mm.decode_predictions(pred)[0][0]

    assert out[1] == "Egyptian_cat"


""" Detection models """


def test_EfficientDetD0_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetD0(pretrained="coco")
    pred = mm(mm.preprocess_input(cat()))
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
    pred = mm(mm.preprocess_input(cat(), input_shape=input_shape))
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_EfficientDetLite1_dynamic_predict():
    mm = keras_cv_attention_models.efficientdet.EfficientDetLite1(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (376, 227, 3)
    pred = mm(mm.preprocess_input(cat(), input_shape=input_shape))
    assert pred.shape == (1, 16641, 94)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_90_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOR_CSP_predict():
    mm = keras_cv_attention_models.yolor.YOLOR_CSP(pretrained="coco")
    pred = mm(mm.preprocess_input(cat()))
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
    pred = mm(mm.preprocess_input(cat(), input_shape=input_shape))
    assert pred.shape == (1, 3330, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOXTiny_predict():
    mm = keras_cv_attention_models.yolox.YOLOXTiny(pretrained="coco")
    pred = mm(mm.preprocess_input(cat()[:, :, ::-1]))
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
    pred = mm(mm.preprocess_input(cat()[:, :, ::-1], input_shape=input_shape))
    assert pred.shape == (1, 1110, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV7_Tiny_predict():
    mm = keras_cv_attention_models.yolov7.YOLOV7_Tiny(pretrained="coco")
    pred = mm(mm.preprocess_input(cat()))
    assert pred.shape == (1, 10647, 85)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV7_Tiny_dynamic_predict():
    mm = keras_cv_attention_models.yolov7.YOLOV7_Tiny(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 276, 3)
    pred = mm(mm.preprocess_input(cat(), input_shape=input_shape))
    assert pred.shape == (1, 3330, 85)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV8_S_predict():
    mm = keras_cv_attention_models.yolov8.YOLOV8_S(pretrained="coco")
    pred = mm(mm.preprocess_input(cat()))
    assert pred.shape == (1, 8400, 144)

    pred_label = mm.decode_predictions(pred)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


def test_YOLOV8_S_dynamic_predict():
    mm = keras_cv_attention_models.yolov8.YOLOV8_S(input_shape=(None, None, 3), pretrained="coco")
    input_shape = (188, 276, 3)
    pred = mm(mm.preprocess_input(cat(), input_shape=input_shape))
    assert pred.shape == (1, 1110, 144)

    pred_label = mm.decode_predictions(pred, input_shape=input_shape)[0][1].numpy()
    assert COCO_80_LABEL_DICT[pred_label[0]] == "cat"


""" Stable Diffusion """


def test_stable_diffusion_no_weights_predict():
    mm = keras_cv_attention_models.stable_diffusion.StableDiffusion(pretrained=None)
    image = keras_cv_attention_models.backend.numpy_image_resize(cat(), [256, 256])
    out = mm("hello world", image=image, init_step=49, strength=0.02, inpaint_mask=[0.5, 0, 1, 1])  # Run only 1 step
    if keras_cv_attention_models.backend.is_torch_backend:
        out = out.detach().cpu().numpy()
        assert out.shape == (1, 3, 256, 256)
    else:
        out = out.numpy()
        assert out.shape == (1, 256, 256, 3)
    assert out.min() > -8 and out.max() < 8  # It should be within this range


""" Segment Anything """


def test_MobileSAM_predict():
    mm = keras_cv_attention_models.segment_anything.MobileSAM()
    points, labels = np.array([(0.5, 0.8)]), np.array([1])
    masks, iou_predictions, low_res_masks = mm(cat(), points, labels)

    assert masks.shape == (4, 512, 512) and iou_predictions.shape == (4,) and low_res_masks.shape == (4, 256, 256)
    assert np.allclose(iou_predictions, np.array([0.98725945, 0.83492416, 0.9997821, 0.96904826]), atol=1e-3)
    assert np.allclose([ii.sum() for ii in masks], [140151, 121550, 139295, 149360], atol=10)
