import pytest
import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append(".")
from keras_cv_attention_models import attention_layers


# Not included: batchnorm_with_activation, conv2d_no_bias, drop_block, hard_swish, layer_norm
def test_anti_alias_downsample():
    input_shape = [2, 28, 28, 192]
    strides = 2
    out = attention_layers.anti_alias_downsample(tf.ones(input_shape), kernel_size=3, strides=strides)
    assert out.shape == [input_shape[0], input_shape[1] // strides, input_shape[2] // strides, input_shape[3]]


def test_BiasLayer():
    aa = attention_layers.BiasLayer()
    input_shape = [2, 14, 14, 192]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_ChannelAffine():
    aa = attention_layers.ChannelAffine()
    input_shape = [2, 14, 14, 192]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_ClassToken():
    aa = attention_layers.ClassToken()
    input_shape = [2, 14 * 14, 192]
    assert aa(tf.ones(input_shape)).shape == [input_shape[0], input_shape[1] + 1, input_shape[2]]


def test_ConvPositionalEncoding():
    aa = attention_layers.ConvPositionalEncoding()
    input_shape = [1, 1 + 14 * 14, 256]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_ConvRelativePositionalEncoding():
    aa = attention_layers.ConvRelativePositionalEncoding()
    input_shape = [1, 8, 1 + 14 * 14, 6]
    assert aa(tf.ones(input_shape), tf.ones(input_shape)).shape == input_shape


def test_cot_attention():
    input_shape = [2, 28, 28, 192]
    assert attention_layers.cot_attention(tf.ones(input_shape), kernel_size=3).shape == input_shape


def test_eca_module():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.eca_module(tf.ones(input_shape))
    assert out.shape == input_shape


def test_halo_attention():
    input_shape = [2, 12, 16, 256]
    out_shape = 384
    out = attention_layers.halo_attention(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_mhsa_with_multi_head_position_and_strides():
    input_shape = [2, 28 * 28, 192]
    strides = 2
    output_dim = 384
    out = attention_layers.mhsa_with_multi_head_position_and_strides(tf.ones(input_shape), output_dim=output_dim, num_heads=4, key_dim=16, strides=strides)
    assert out.shape == [input_shape[0], input_shape[1] // strides // strides, output_dim]


def test_mhsa_with_relative_position_embedding():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.mhsa_with_relative_position_embedding(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_mixer_block():
    input_shape = [2, 28 * 28, 192]
    out = attention_layers.mixer_block(tf.ones(input_shape), tokens_mlp_dim=14 * 14, channels_mlp_dim=9 * 4)
    assert out.shape == input_shape


def test_mlp_block():
    input_shape = [2, 28 * 28, 192]
    out = attention_layers.mlp_block(tf.ones(input_shape), hidden_dim=9 * 4)
    assert out.shape == input_shape


def test_MultiHeadPositionalEmbedding():
    aa = attention_layers.MultiHeadPositionalEmbedding()
    input_shape = [2, 8, 16, 49]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_outlook_attention():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.outlook_attention(tf.ones(input_shape), embed_dim=192, num_head=4)
    assert out.shape == input_shape


def test_outlook_attention_simple():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.outlook_attention_simple(tf.ones(input_shape), embed_dim=192, num_head=4)
    assert out.shape == input_shape


def test_PositionalEmbedding():
    aa = attention_layers.PositionalEmbedding()
    input_shape = [2, 8, 16, 49]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_RelativePositionalEmbedding():
    aa = attention_layers.RelativePositionalEmbedding()
    hh = pos_hh = 14
    ww = pos_ww = 16
    input_shape = [2, 4, hh, ww, 32]
    assert aa(tf.ones(input_shape)).shape == [input_shape[0], input_shape[1], hh, ww, pos_hh, pos_ww]


def test_rsoftmax():
    input_shape = [2, 1, 1, 49 * 2]
    out = attention_layers.rsoftmax(tf.ones(input_shape), groups=2)
    assert out.shape == input_shape


def test_ScaledStandardizedConv2D():
    filters = 64
    aa = attention_layers.ScaledStandardizedConv2D(filters=filters, kernel_size=3, padding="SAME")
    input_shape = [2, 28, 28, 32]
    assert aa(tf.ones(input_shape)).shape == [*input_shape[:3], filters]


def test_se_module():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.se_module(tf.ones(input_shape), se_ratio=0.25)
    assert out.shape == input_shape


def test_spatial_gating_block():
    input_shape = [2, 28 * 28, 192]
    out = attention_layers.spatial_gating_block(tf.ones(input_shape))
    assert out.shape == [*input_shape[:2], input_shape[-1] // 2]


def test_split_attention_conv2d():
    input_shape = [2, 28, 28, 192]
    filters = 384
    out = attention_layers.split_attention_conv2d(tf.ones(input_shape), filters=filters)
    assert out.shape == [*input_shape[:3], filters]


def test_tpu_extract_patches_overlap_1():
    inputs = tf.random.uniform([1, 64, 27, 192])
    pad_inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
    pathes = tf.image.extract_patches(pad_inputs, [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')
    tpu_pathes = attention_layers.tpu_extract_patches_overlap_1(pad_inputs, 3, 2, padding='VALID')
    tf.assert_less(tf.abs(pathes - tpu_pathes), 1e-7)


def test_ZeroInitGain():
    aa = attention_layers.ZeroInitGain()
    input_shape = [2, 28, 28, 32]
    assert aa(tf.ones(input_shape)).shape == input_shape
