from keras_cv_attention_models.model_surgery.model_surgery import (
    SAMModel,
    DropConnect,
    add_l2_regularizer_2_model,
    align_pyramide_feature_output_by_image_data_format,
    change_model_input_shape,
    convert_to_dynamic_input_shape,
    convert_dense_to_conv,
    convert_extract_patches_to_conv,
    convert_gelu_to_approximate,
    convert_gelu_and_extract_patches_for_tflite,  # [Deprecated], use convert_gelu_to_approximate -> convert_extract_patches_to_conv instead
    convert_groups_conv2d_2_split_conv2d,
    convert_to_mixed_float16,
    convert_mixed_float16_to_float32,
    convert_to_fixed_batch_size,
    convert_to_fused_conv_bn_model,
    convert_to_token_label_model,
    convert_layers_to_deploy_inplace,
    count_params,
    export_onnx,
    fuse_sequential_conv_strict,
    fuse_channel_affine_to_conv_dense,
    fuse_reparam_blocks,
    fuse_distill_head,
    get_actual_survival_probabilities,
    get_actual_drop_connect_rates,
    get_flops,
    get_global_avg_pool_layer_id,
    get_pyramide_feature_layers,
    prepare_for_tflite,
    remove_layer_single_input,
    replace_ReLU,
    replace_add_with_drop_connect,
    replace_add_with_stochastic_depth,
    replace_stochastic_depth_with_add,
    split_model_to_head_body_tail_by_blocks,
    swin_convert_pos_emb_mlp_to_MlpPairwisePositionalEmbedding_weights,
)
