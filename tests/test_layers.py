import pytest
import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append(".")
from keras_cv_attention_models import attention_layers


# Not included: batchnorm_with_activation, conv2d_no_bias, drop_block, hard_swish, phish, mish, layer_norm
def test_add_pre_post_process_tf():
    input_shape = (224, 224, 3)
    fake_input = tf.random.uniform([1, *input_shape]) * 255
    mm = keras.models.Sequential()
    rescale_mode = "tf"

    attention_layers.add_pre_post_process(mm, rescale_mode=rescale_mode, input_shape=input_shape)
    aa = mm.preprocess_input(fake_input)
    bb = keras.applications.imagenet_utils.preprocess_input(fake_input, mode=rescale_mode)
    tf.assert_less(tf.abs(aa - bb), 1e-6)

    aa = mm.preprocess_input(fake_input[0])
    bb = keras.applications.imagenet_utils.preprocess_input(fake_input, mode=rescale_mode)
    tf.assert_less(tf.abs(aa - bb), 1e-6)


def test_add_pre_post_process_torch():
    input_shape = (224, 224, 3)
    fake_input = tf.random.uniform([1, *input_shape]) * 255
    mm = keras.models.Sequential()
    rescale_mode = "torch"

    attention_layers.add_pre_post_process(mm, rescale_mode=rescale_mode, input_shape=input_shape)
    aa = mm.preprocess_input(fake_input)
    bb = keras.applications.imagenet_utils.preprocess_input(fake_input, mode=rescale_mode)
    tf.assert_less(tf.abs(aa - bb), 1e-6)

    aa = mm.preprocess_input(fake_input[0])
    bb = keras.applications.imagenet_utils.preprocess_input(fake_input, mode=rescale_mode)
    tf.assert_less(tf.abs(aa - bb), 1e-6)


def test_add_pre_post_process_raw():
    input_shape = (224, 224, 3)
    fake_input = tf.random.uniform([1, *input_shape]) * 255
    mm = keras.models.Sequential()
    rescale_mode = "raw"

    attention_layers.add_pre_post_process(mm, rescale_mode=rescale_mode, input_shape=input_shape)
    aa = mm.preprocess_input(fake_input)
    tf.assert_less(tf.abs(aa - fake_input), 1e-6)


def test_anti_alias_downsample():
    input_shape = [2, 28, 28, 192]
    strides = 2
    out = attention_layers.anti_alias_downsample(tf.ones(input_shape), kernel_size=3, strides=strides)
    assert out.shape == [input_shape[0], input_shape[1] // strides, input_shape[2] // strides, input_shape[3]]


def test_BiasLayer():
    aa = attention_layers.BiasLayer()
    input_shape = [2, 14, 14, 192]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_BiasPositionalEmbedding():
    aa = attention_layers.BiasPositionalEmbedding()
    input_shape = [2, 4, 36, 9]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_ChannelAffine():
    aa = attention_layers.ChannelAffine()
    input_shape = [2, 14, 14, 192]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_ComplexDense():
    aa = attention_layers.ComplexDense()
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


def test_cross_covariance_attention():
    input_shape = [2, 14, 16, 192]
    assert attention_layers.cross_covariance_attention(tf.ones(input_shape)).shape == input_shape


def test_eca_module():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.eca_module(tf.ones(input_shape))
    assert out.shape == input_shape


def test_EvoNormalization():
    input_shape = [2, 32, 17, 44]
    out = attention_layers.EvoNormalization()(tf.ones(input_shape))
    assert out.shape == input_shape

    out = attention_layers.EvoNormalization(data_format="channels_first")(tf.ones(input_shape))
    assert out.shape == input_shape

    out = attention_layers.EvoNormalization(num_groups=24)(tf.ones(input_shape))
    assert out.shape == input_shape

    out = attention_layers.EvoNormalization(data_format="channels_first", num_groups=24)(tf.ones(input_shape))
    assert out.shape == input_shape


def test_ExpLogitScale():
    aa = attention_layers.ExpLogitScale(axis=[1, 2])
    input_shape = [1, 32, 32, 192]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_fold_by_conv2d_transpose():
    inputs = tf.random.uniform([1, 64, 27, 192])
    pad_inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
    pathes = attention_layers.CompatibleExtractPatches(sizes=3, strides=2, rates=1, padding="VALID")(pad_inputs)
    reverse = attention_layers.fold_by_conv2d_transpose(pathes, inputs.shape[1:], kernel_size=3, strides=2, padding="SAME")
    assert reverse.shape == inputs.shape


def test_global_local_filter():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.global_local_filter(tf.ones(input_shape))
    assert out.shape == input_shape


def test_global_response_normalize():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.global_response_normalize(tf.ones(input_shape))
    assert out.shape == input_shape


def test_gnconv():
    input_shape = [2, 28, 28, 192]
    out_1 = attention_layers.gnconv(tf.ones(input_shape), use_global_local_filter=False)
    assert out_1.shape == input_shape
    out_2 = attention_layers.gnconv(tf.ones(input_shape), use_global_local_filter=True)
    assert out_2.shape == input_shape


def test_halo_attention():
    input_shape = [2, 12, 16, 256]
    out_shape = 384
    out = attention_layers.halo_attention(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_linear_self_attention():
    input_shape = [2, 14, 16, 256]
    out = attention_layers.linear_self_attention(tf.ones(input_shape), attn_axis=-1)
    assert out.shape == input_shape


def test_light_mhsa_with_multi_head_relative_position_embedding():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.light_mhsa_with_multi_head_relative_position_embedding(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_mhsa_with_multi_head_position_and_strides():
    input_shape = [2, 28, 18, 192]
    strides = 2
    output_dim = 384
    out = attention_layers.mhsa_with_multi_head_position_and_strides(tf.ones(input_shape), output_dim=output_dim, num_heads=4, key_dim=16, strides=strides)
    assert out.shape == [input_shape[0], input_shape[1] // strides, input_shape[2] // strides, output_dim]


def test_mhsa_with_relative_position_embedding():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.mhsa_with_relative_position_embedding(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_mhsa_with_multi_head_relative_position_embedding():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.mhsa_with_multi_head_relative_position_embedding(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_mlp_mixer_block():
    input_shape = [2, 28 * 28, 192]
    out = attention_layers.mlp_mixer_block(tf.ones(input_shape), tokens_mlp_dim=14 * 14, channels_mlp_dim=9 * 4)
    assert out.shape == input_shape


def test_mlp_block():
    input_shape = [2, 28 * 28, 192]
    out = attention_layers.mlp_block(tf.ones(input_shape), hidden_dim=9 * 4)
    assert out.shape == input_shape


def test_multi_head_self_attention():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.multi_head_self_attention(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_multi_head_self_attention_channel():
    input_shape = [2, 14, 16, 256]
    out_shape = 384
    out = attention_layers.multi_head_self_attention_channel(tf.ones(input_shape), num_heads=4, out_shape=out_shape)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_shape]


def test_MultiHeadPositionalEmbedding():
    aa = attention_layers.MultiHeadPositionalEmbedding()
    input_shape = [2, 8, 16, 49]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_MultiHeadRelativePositionalEmbedding():
    aa = attention_layers.MultiHeadRelativePositionalEmbedding()
    input_shape = [2, 8, 29 * 29 + 1, 29 * 29 + 1]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_MultiHeadRelativePositionalKernelBias():
    aa = attention_layers.MultiHeadRelativePositionalKernelBias()
    input_shape = [2, 29 * 29, 8, 3, 5 * 5]
    assert aa(tf.ones(input_shape)).shape == input_shape

    aa = attention_layers.MultiHeadRelativePositionalKernelBias(input_height=19, is_heads_first=True)
    input_shape = [2, 8, 19 * 29, 5 * 5]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_neighborhood_attention():
    input_shape = [2, 28, 32, 192]
    out = attention_layers.neighborhood_attention(tf.ones(input_shape))
    assert out.shape == input_shape


def test_outlook_attention():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.outlook_attention(tf.ones(input_shape), embed_dim=192, num_heads=4)
    assert out.shape == input_shape


def test_outlook_attention_simple():
    input_shape = [2, 28, 28, 192]
    out = attention_layers.outlook_attention_simple(tf.ones(input_shape), embed_dim=192, num_heads=4)
    assert out.shape == input_shape


def test_PairWiseRelativePositionalEmbedding():
    aa = attention_layers.PairWiseRelativePositionalEmbedding()
    window_patch, window_height, window_width, channel = 9, 5, 3, 23
    input_shape = [window_patch, window_height, window_width, channel]
    coords_shape = [(2 * window_height - 1) * (2 * window_width - 1), 2]
    pos_emb_shape = [window_height * window_width, window_height * window_width, 2]
    relative_log_coords = aa(tf.ones(input_shape))
    relative_position_bias = attention_layers.PairWiseRelativePositionalEmbeddingGather(window_height=window_height)(relative_log_coords)
    assert relative_log_coords.shape == coords_shape
    assert relative_position_bias.shape == pos_emb_shape


def test_phase_aware_token_mixing():
    input_shape = [2, 28, 28, 192]
    out_channel = 384
    out = attention_layers.phase_aware_token_mixing(tf.ones(input_shape), out_channel=out_channel)
    assert out.shape == [input_shape[0], input_shape[1], input_shape[2], out_channel]


def test_PositionalEmbedding():
    aa = attention_layers.PositionalEmbedding()
    input_shape = [2, 8, 16, 49]
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_PositionalEncodingFourier():
    aa = attention_layers.PositionalEncodingFourier()
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


def test_shifted_window_attention():
    input_shape = [2, 28, 32, 192]
    out = attention_layers.shifted_window_attention(tf.ones(input_shape), window_size=7, num_heads=4, shift_size=0.5)
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


def test_CompatibleExtractPatches():
    inputs = tf.random.uniform([1, 64, 27, 192])
    pad_inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
    pathes = tf.image.extract_patches(pad_inputs, [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
    patches_by_conv2d = attention_layers.CompatibleExtractPatches(sizes=3, strides=2, rates=1, padding="SAME")(inputs)
    tf.assert_less(tf.abs(pathes - patches_by_conv2d), 1e-7)

    patches_1 = attention_layers.CompatibleExtractPatches(sizes=3, strides=2, rates=1, padding="SAME", compressed=False)(inputs)
    patches_tpu_1 = attention_layers.CompatibleExtractPatches(3, 2, 1, padding="SAME", compressed=False, force_conv=True)(inputs)
    tf.assert_less(tf.abs(patches_1 - patches_tpu_1), 1e-7)

    patches_2 = attention_layers.CompatibleExtractPatches(sizes=3, strides=2, rates=1, padding="SAME", compressed=True)(inputs)
    patches_tpu_2 = attention_layers.CompatibleExtractPatches(3, 2, 1, padding="SAME", compressed=True, force_conv=True)(inputs)
    tf.assert_less(tf.abs(patches_2 - patches_tpu_2), 1e-7)


def test_window_attention():
    input_shape = [2, 28, 32, 192]
    out = attention_layers.window_attention(tf.ones(input_shape), window_size=7, num_heads=4)
    assert out.shape == input_shape


def test_WindowAttentionMask():
    height, width, window_height, window_width, shift_height, shift_width = 36, 48, 6, 6, 3, 3
    num_heads, query_blocks = 8, window_height * window_width
    input_shape = [1 * (height // window_height) * (width // window_width), num_heads, query_blocks, query_blocks]

    aa = attention_layers.WindowAttentionMask(height, width, window_height, window_width, shift_height, shift_width)
    assert aa(tf.ones(input_shape)).shape == input_shape


def test_window_mhsa_with_pair_wise_positional_embedding():
    input_shape = [4, 7, 8, 256]
    out = attention_layers.window_mhsa_with_pair_wise_positional_embedding(tf.ones(input_shape), num_heads=4)
    assert out.shape == input_shape


def test_ZeroInitGain():
    aa = attention_layers.ZeroInitGain()
    input_shape = [2, 28, 28, 32]
    assert aa(tf.ones(input_shape)).shape == input_shape
