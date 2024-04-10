import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers

""" Convert and replacing """


class SAMModel(models.Model):
    """
    Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf)
    Implementation by: [Keras SAM (Sharpness-Aware Minimization)](https://qiita.com/T-STAR/items/8c3afe3a116a8fc08429)

    Usage is same with `keras.models.Model`: `model = SAMModel(inputs, outputs, rho=sam_rho, name=name)`
    """

    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        import tensorflow as tf

        self.rho = tf.constant(rho, dtype=tf.float32)
        self.tf = tf

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # 1st step
        with self.tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        norm = self.tf.linalg.global_norm(gradients)
        scale = self.rho / (norm + 1e-12)
        e_w_list = []
        for v, grad in zip(trainable_vars, gradients):
            e_w = grad * scale
            v.assign_add(e_w)
            e_w_list.append(e_w)

        # 2nd step
        with self.tf.GradientTape() as tape:
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, sample_weight=sample_weight, regularization_losses=self.losses)
        gradients_adv = tape.gradient(loss_adv, trainable_vars)
        for v, e_w in zip(trainable_vars, e_w_list):
            v.assign_sub(e_w)

        # optimize
        self.optimizer.apply_gradients(zip(gradients_adv, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


@backend.register_keras_serializable(package="model_surgery")
class DropConnect(layers.Layer):
    def __init__(self, rate=0, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.rate = rate
        self.supports_masking = False

    def build(self, input_shape):
        if self.rate > 0:
            noise_shape = [None] + [1] * (len(input_shape) - 1)  # [None, 1, 1, 1]
            self.drop = layers.Dropout(self.rate, noise_shape=noise_shape, name=self.name + "drop")
        else:
            self.drop = lambda xx: xx

    def call(self, inputs, **kwargs):
        shortcut, deep = inputs
        deep = self.drop(deep)
        return layers.Add()([shortcut, deep])

    def get_config(self):
        config = super(DropConnect, self).get_config()
        config.update({"rate": self.rate})
        return config


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=False, apply_to_bias=False):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    from tensorflow import keras

    if 0:
        regularizers_type = {}
        for layer in model.layers:
            rrs = [kk for kk in layer.__dict__.keys() if "regularizer" in kk and not kk.startswith("_")]
            if len(rrs) != 0:
                # print(layer.name, layer.__class__.__name__, rrs)
                if layer.__class__.__name__ not in regularizers_type:
                    regularizers_type[layer.__class__.__name__] = rrs
        print(regularizers_type)

    for layer in model.layers:
        attrs = []
        if isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv2D):
            # print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["kernel_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return models.clone_model(model)


def replace_ReLU(model, target_activation="PReLU", **kwargs):
    from tensorflow import keras
    from tensorflow.keras.layers import ReLU, PReLU, Activation

    def convert_ReLU(layer):
        # print(layer.name)
        if isinstance(layer, ReLU) or (isinstance(layer, Activation) and layer.activation == keras.activations.relu):
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                # Default initial value in mxnet and pytorch is 0.25
                return PReLU(shared_axes=[1, 2], alpha_initializer=initializers.Constant(0.25), name=layer_name, **kwargs)
            elif isinstance(target_activation, str):
                layer_name = layer.name.replace("_relu", "_" + target_activation)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return Activation(activation=target_activation, name=layer_name, **kwargs)
            else:
                act_class_name = target_activation.__name__
                layer_name = layer.name.replace("_relu", "_" + act_class_name)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return target_activation(**kwargs)
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=convert_ReLU)


def change_model_input_shape(model, new_input_shape):
    import json
    import os

    if len(new_input_shape) == len(model.input_shape) - 1:
        new_input_shape = [model.input_shape[0], *new_input_shape]
    elif len(new_input_shape) == len(model.input_shape) - 2:
        new_input_shape = [model.input_shape[0], *new_input_shape, model.input_shape[-1]]
    else:
        new_input_shape = list(new_input_shape)

    if list(model.input_shape) == new_input_shape:
        return model

    aa = json.loads(model.to_json())
    aa["config"]["layers"][0]["config"]["batch_input_shape"] = new_input_shape
    bb = models.model_from_json(json.dumps(aa))
    temp_name = "__" + model.name + "_change_model_input_shape_temp__.h5"
    model.save_weights(temp_name)
    bb.load_weights(temp_name)
    os.remove(temp_name)
    print(">>>> Changed model input shape from {} to {}".format(model.input_shape, bb.input_shape))
    return bb


def convert_to_dynamic_input_shape(model):
    if not backend.is_tensorflow_backend:
        return model  # Do nothing for pytorch model

    def __convert_to_dynamic_input_shape__(layer):
        if hasattr(layer, "load_resized_weights"):
            aa = layer.__class__.from_config(layer.get_config())
            # aa.build(layer.input_shape)  # [???] Error when saving ValueError: Unable to create dataset (name already exists)
            _ = aa(initializers.zeros()([1, *layer.input_shape[1:]]))
            aa.set_weights(layer.get_weights())
            return aa
        else:
            return layer

    input_shape = [None] * (len(model.input_shape) - 2) + list(model.input_shape)[-1:]
    bb = models.clone_model(model, layers.Input(input_shape), clone_function=__convert_to_dynamic_input_shape__)
    print(">>>> Changed model input shape from {} to {}".format(model.input_shape, bb.input_shape))
    return bb


# Not using
def find_layer_name_nested(layers, layer_name_cond, reverse=False):
    layers = layers[::-1] if reverse else layers
    for ii in layers:
        if layer_name_cond(ii.name):
            return ii
        if hasattr(ii, "layers"):  # nested
            found_layer = find_layer_name_nested(ii.layers, layer_name_cond, reverse=reverse)
            if found_layer is not None:
                return ii  # Return nested block
    return None


def split_model_to_head_body_tail_by_blocks(model):
    for ii in model.layers:
        if "stack" in ii.name or "block" in ii.name:
            body_start_layer = ii
            break
    assert body_start_layer is not None, "None layer with name containing 'stack' or 'block' found"

    for ii in model.layers[::-1]:
        if "stack" in ii.name or "block" in ii.name:
            body_end_layer = ii
            break
    print(">>>> Split model to head + body + tail by blocks, body_start_layer: {}, body_end_layer: {}".format(body_start_layer.name, body_end_layer.name))

    head_model = models.Model(model.inputs[0], body_start_layer.input, name=model.name + "_head")
    body_model = models.Model(body_start_layer.input, body_end_layer.output, name=model.name + "_body")
    tail_model = models.Model(body_end_layer.output, model.outputs[0], name=model.name + "_tail")
    return head_model, body_model, tail_model


def replace_add_with_stochastic_depth(model, survivals=(1, 0.8)):
    """
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
    """
    from tensorflow_addons.layers import StochasticDepth

    add_layers = [ii.name for ii in model.layers if isinstance(ii, layers.Add)]
    total_adds = len(add_layers)
    if isinstance(survivals, float):
        survivals = [survivals] * total_adds
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start - (1 - end) * float(ii) / total_adds for ii in range(total_adds)]
    survivals_dict = dict(zip(add_layers, survivals))

    def __replace_add_with_stochastic_depth__(layer):
        if isinstance(layer, layers.Add):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_add", "_stochastic_depth")
            new_layer_name = new_layer_name.replace("add_", "stochastic_depth_")
            survival_probability = survivals_dict[layer_name]
            if survival_probability < 1:
                print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival_probability)
                return StochasticDepth(survival_probability, name=new_layer_name)
            else:
                return layer
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_add_with_stochastic_depth__)


def replace_add_with_drop_connect(model, drop_rate=(0, 0.2)):
    """
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
    """

    add_layers = [ii.name for ii in model.layers if isinstance(ii, layers.Add)]
    total_adds = len(add_layers)
    if isinstance(drop_rate, float):
        drop_rates = [drop_rate] * total_adds
    elif isinstance(drop_rate, (list, tuple)) and len(drop_rate) == 2:
        start, end = drop_rate
        drop_rates = [(end - start) * float(ii) / total_adds for ii in range(total_adds)]
    drop_conn_rate_dict = dict(zip(add_layers, drop_rates))

    def __replace_add_with_stochastic_depth__(layer):
        if isinstance(layer, layers.Add):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_add", "_drop_conn")
            new_layer_name = new_layer_name.replace("add_", "drop_conn_")
            drop_conn_rate = drop_conn_rate_dict[layer_name]
            if drop_conn_rate < 1:
                print("Converting:", layer_name, "-->", new_layer_name, ", drop_conn_rate:", drop_conn_rate)
                return DropConnect(drop_conn_rate, name=new_layer_name)
            else:
                return layer
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_add_with_stochastic_depth__)


def replace_stochastic_depth_with_add(model, drop_survival=False):
    from tensorflow_addons.layers import StochasticDepth

    def __replace_stochastic_depth_with_add__(layer):
        if isinstance(layer, StochasticDepth):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_stochastic_depth", "_lambda")
            survival = layer.survival_probability
            print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival)
            if drop_survival or not survival < 1:
                return layers.Add(name=new_layer_name)
            else:
                return layers.Lambda(lambda xx: xx[0] + xx[1] * survival, name=new_layer_name)
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_stochastic_depth_with_add__)


def convert_to_token_label_model(model, pool_layer_id="auto"):
    # Search pool layer id
    num_total_layers = len(model.layers)
    if pool_layer_id == "auto":
        for header_layer_id, layer in enumerate(model.layers[::-1]):
            header_layer_id = num_total_layers - header_layer_id - 1
            print("[Search pool layer] header_layer_id = {}, layer.name = {}".format(header_layer_id, layer.name))
            if isinstance(layer, layers.GlobalAveragePooling2D):
                break
        pool_layer_id = header_layer_id

    nn = model.layers[pool_layer_id - 1].output  # layer output before pool layer

    # Add header layers w/o pool layer
    for header_layer_id in range(pool_layer_id + 1, num_total_layers):
        aa = model.layers[header_layer_id]
        config = aa.get_config()
        config["name"] = config["name"] + "_token_label"
        if isinstance(aa, layers.LayerNormalization) and config["axis"] == [1]:
            config["axis"] = [-1]
        # print(config)
        print("[Build new layer] header_layer_id = {}, layer.name = {}".format(header_layer_id, config["name"]))

        bb = aa.__class__.from_config(config)
        bb.build(nn.shape)
        bb.set_weights(aa.get_weights())
        nn = bb(nn)
    token_label_model = models.Model(model.inputs[0], [*model.outputs, nn])
    print("token_label_model.output_shape =", token_label_model.output_shape)
    return token_label_model


""" Get model info """


def get_actual_survival_probabilities(model):
    from tensorflow_addons.layers import StochasticDepth

    return [ii.survival_probability for ii in model.layers if isinstance(ii, StochasticDepth)]


def get_actual_drop_connect_rates(model):
    return [ii.rate for ii in model.layers if isinstance(ii, layers.Dropout) or isinstance(ii, DropConnect)]


def get_pyramide_feature_layers(model, match_reg="^stack_?(\\d+).*output.*$"):
    """Pick all stack output layers"""
    import re

    if hasattr(model, "extract_features"):
        return model.extract_features()

    dd = {}
    pre_stack, pre_output_shape = 0, model.input_shape[1:]
    for ii in model.layers:
        cur_name = ii.name
        # if cur_name.startswith("pre_output") and ii.output_shape[1:] == pre_output_shape:
        #     cur_name = "stack_{}".format(pre_stack) + cur_name  # For Swin

        matched = re.match(match_reg, cur_name)
        if matched is not None:
            cur_stack = "stack_" + matched[1] + "_output"
            pre_stack, pre_output_shape = matched[1], ii.output_shape[1:]
            dd.update({cur_stack: ii})

    """ Filter those have same downsample rate """
    ee = {str(vv.output_shape[2]): vv for kk, vv in dd.items()}
    return list(ee.values())


def align_pyramide_feature_output_by_image_data_format(pyramide_feature_layers):
    aa = [ii.output_shape for ii in pyramide_feature_layers]
    bb = [ii[1] for ii in aa]
    if all([ii is None or bb[id - 1] is None or ii <= bb[id - 1] for id, ii in enumerate(bb[1:], start=1)]):
        features_data_format = "channels_last"
    else:
        features_data_format = "channels_first"

    if features_data_format == backend.image_data_format():
        output_names = [ii.name for ii in pyramide_feature_layers]
        outputs = [ii.output for ii in pyramide_feature_layers]
    elif features_data_format == "channels_last":  # backend.image_data_format is "channels_first"
        output_names = [ii.name + "_perm" for ii in pyramide_feature_layers]
        outputs = [layers.Permute([3, 1, 2], name=ii.name + "_perm")(ii.output) for ii in pyramide_feature_layers]
    else:  # features_data_format is "channels_first" and backend.image_data_format is "channels_last". Should not happen
        output_names = [ii.name + "_perm" for ii in pyramide_feature_layers]
        outputs = [layers.Permute([2, 3, 1], name=ii.name + "_perm")(ii.output) for ii in pyramide_feature_layers]
    return output_names, outputs


def get_global_avg_pool_layer_id(model):
    """Search GlobalAveragePooling2D layer id"""
    num_total_layers = len(model.layers)
    for header_layer_id, layer in enumerate(model.layers[::-1]):
        header_layer_id = num_total_layers - header_layer_id - 1
        print("[Search pool layer] header_layer_id = {}, layer.name = {}".format(header_layer_id, layer.name))
        if isinstance(layer, layers.GlobalAveragePooling2D):
            break
        if hasattr(layer, "function") and hasattr(layer.function, "__name__") and layer.function.__name__ == "reduce_mean":
            break
    return header_layer_id


def get_layer_as_inputs_count_from_mode_json(model):
    # dd = {}
    # for ii in cc.layers:
    #     layer_inputs = ii.input if isinstance(ii.input, (list, tuple)) else [ii.input]
    #     for layer_input in layer_inputs:
    #         dd[layer_input.node.layer.name] = dd.get(layer_input.node.layer.name, 0) + 1
    if isinstance(model, dict):
        model_config = model
    else:
        import json

        model_config = json.loads(model.to_json())

    dd = {}
    is_inbound_elem = lambda xx: isinstance(xx, list) and isinstance(xx[0], str)
    for layer in model_config["config"]["layers"]:
        # print(layer['inbound_nodes'])
        for ii in layer["inbound_nodes"]:
            # print(ii)
            # single input layer: [[['stack4_block2_1_conv', 0, 0, {}]]]
            # multi input layer: 'inbound_nodes': [['stack4_block2_1_conv', 0, 0, {'y': ['stack4_block2_1_short', 0, 0], 'name': None}]]}
            if is_inbound_elem(ii):
                dd[ii[0]] = dd.get(ii[0], 0) + 1
                for kk, vv in ii[3].items():
                    if is_inbound_elem(vv):
                        dd[vv[0]] = dd.get(vv[0], 0) + 1
            elif isinstance(ii, list) and isinstance(ii[0], list):
                for jj in ii:
                    dd[jj[0]] = dd.get(jj[0], 0) + 1
                    for kk, vv in jj[3].items():
                        if is_inbound_elem(vv):
                            dd[vv[0]] = dd.get(vv[0], 0) + 1
    return dd


def get_flops(model, input_shape=None):
    input_shape = [(1, *ii.shape[1:]) for ii in model.inputs] if input_shape is None else input_shape
    if backend.is_tensorflow_backend:
        # https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-849439287
        import tensorflow as tf
        from tensorflow.python.profiler import model_analyzer, option_builder

        input_signature = [tf.TensorSpec(shape=ss, dtype=ii.dtype, name=ii.name) for ii, ss in zip(model.inputs, input_shape)]
        forward_graph = tf.function(model, [input_signature]).get_concrete_function().graph
        options = option_builder.ProfileOptionBuilder.float_operation()
        graph_info = model_analyzer.profile(forward_graph, options=options)
        flops = graph_info.total_float_ops // 2
    else:
        import thop
        import torch

        inputs = [torch.ones(ii) for ii in input_shape]
        flops, params = thop.profile(model, inputs=tuple([inputs]))
    print(">>>> FLOPs: {:,}, GFLOPs: {:.4f}G".format(flops, flops / 1e9))
    return flops


def count_params(model):
    # "parameters" and "requires_grad" are for torch model, while "weights" and "trainable" for TF models
    total_params, trainable_params = 0, 0
    for ii in model.parameters() if hasattr(model, "parameters") else model.weights:
        cur_params = np.prod(ii.shape)
        total_params += cur_params
        trainable = ii.requires_grad if hasattr(ii, "requires_grad") else ii.trainable
        trainable_params += cur_params if trainable else 0
    non_trainable_params = total_params - trainable_params
    print("Total params: {:,} | Trainable params: {:,} | Non-trainable params:{:,}".format(total_params, trainable_params, non_trainable_params))
    return total_params, trainable_params


""" Inference and deploy """


def convert_to_mixed_float16(model, convert_batch_norm=False, policy_name="mixed_float16"):
    from tensorflow import keras
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear

    policy = keras.mixed_precision.Policy(policy_name)
    policy_config = keras.utils.serialize_keras_object(policy)

    def do_convert_to_mixed_float16(layer):
        if not convert_batch_norm and isinstance(layer, layers.BatchNormalization):
            return layer
        if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
            aa = layer.get_config()
            aa.update({"dtype": policy_config})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights())
            return bb
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)


def convert_mixed_float16_to_float32(model):
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear

    def do_convert_to_mixed_float16(layer):
        if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
            aa = layer.get_config()
            aa.update({"dtype": "float32"})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights())
            return bb
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)


def fuse_layer_single_input(model, fuse_layer_condition, pre_layer_condition=None, layer_config_process=None, layer_weight_process=None, verbose=0):
    import json

    pre_layer_condition = (lambda layer: True) if pre_layer_condition is None else pre_layer_condition

    """ Check bn layers with conv layer input """
    model_config = json.loads(model.to_json())
    ee = {layer["name"]: layer for layer in model_config["config"]["layers"]}
    pre_fused_layers, fused_layers = [], []
    for layer in model_config["config"]["layers"]:
        if fuse_layer_condition(layer) and len(layer["inbound_nodes"]) == 1:
            input_node = layer["inbound_nodes"][0][0]
            if isinstance(input_node, list) and pre_layer_condition(ee.get(input_node[0], {})):
                pre_fused_layers.append(input_node[0])
                fused_layers.append(layer["name"])
    if verbose > 0:
        print(">>>> pre_fused_layers =", pre_fused_layers, "fused_layers =", fused_layers)
        print(">>>> len(pre_fused_layers) =", len(pre_fused_layers), "len(fused_layers) =", len(fused_layers))
        print()
        # len(fuse_convs) = 53, len(fuse_bns) = 53

    """ Create new model config """
    layers = []
    fused_layers_dict = dict(zip(fused_layers, pre_fused_layers))
    pre_fused_layers_dict = dict(zip(pre_fused_layers, fused_layers))
    is_inbound_elem = lambda xx: isinstance(xx, list) and isinstance(xx[0], str)
    for layer in model_config["config"]["layers"]:
        if layer["name"] in pre_fused_layers and layer_config_process is not None:
            if verbose > 0:
                print(">>>> Create new layer config for:", layer["name"])
            fused_layer_config = ee.get(pre_fused_layers_dict[layer["name"]])["config"]
            layer["config"] = layer_config_process(layer["config"], fused_layer_config)
        elif layer["name"] in fused_layers:
            if verbose > 0:
                print(">>>> Remove layer:", layer["name"])
            continue

        for ii in layer["inbound_nodes"]:
            # print(ii)
            # single input layer: [[['stack4_block2_1_conv', 0, 0, {}]]]
            # multi input layer: 'inbound_nodes': [['stack4_block2_1_conv', 0, 0, {'y': ['stack4_block2_1_short', 0, 0], 'name': None}]]}
            # multi_input layer: 'inbound_nodes': [[['stack1_block1_1_conv', 0, 0, {}], ['stem_1_conv', 0, 0, {}], ['stem_2_conv', 0, 0, {}]]]}
            if is_inbound_elem(ii):
                # print(">>>> Replace inbound_nodes: {}, {} --> {}".format(layer["name"], ii[0], fused_bn_dict[ii[0]]))
                ii[0] = fused_layers_dict.get(ii[0], ii[0])
                ii[3] = {kk: [fused_layers_dict.get(vv[0], vv[0]), *vv[1:]] if is_inbound_elem(vv) else vv for kk, vv in ii[3].items()}
            elif isinstance(ii, list) and isinstance(ii[0], list):
                for jj in ii:
                    jj[0] = fused_layers_dict.get(jj[0], jj[0])
                    jj[3] = {kk: [fused_layers_dict.get(vv[0], vv[0]), *vv[1:]] if is_inbound_elem(vv) else vv for kk, vv in jj[3].items()}

        layers.append(layer)
    model_config["config"]["layers"] = layers
    new_model = models.model_from_json(json.dumps(model_config))
    if verbose > 0:
        print()

    """ New model set layer weights by layer names """
    for layer in new_model.layers:
        if layer.name in fused_layers:  # This should not happen
            continue

        orign_layer = model.get_layer(layer.name)
        if layer.name in pre_fused_layers_dict and layer_weight_process is not None:
            orign_fused_layer = model.get_layer(pre_fused_layers_dict[layer.name])
            if verbose > 0:
                print(">>>> Fuse weights {} <- {}".format(layer.name, orign_fused_layer.name))
            layer.set_weights(layer_weight_process(orign_layer, orign_fused_layer))
        else:
            layer.set_weights(orign_layer.get_weights())
    return new_model


def fuse_bn_dense_weights(bn_layer, dense_layer):
    """
    https://github.com/THU-MIG/RepViT/blob/main/model/repvit.py#L168
    Convert model by fusing batchnorm + dense

    Exampls:
    >>> from keras_cv_attention_models.model_surgery import model_surgery
    >>> ii = tf.random.uniform([1, 32, 32, 3])
    >>> mm = keras.models.Sequential([keras.layers.Input([32, 32, 3]), keras.layers.BatchNormalization(), keras.layers.Dense(10)])
    >>> aa = keras.layers.Dense(10)
    >>> aa.build([None, 32, 32, 3])
    >>> aa.set_weights(model_surgery.fuse_bn_dense_weights(mm.layers[0], mm.layers[1]))
    >>> print(f"{np.allclose(mm(ii), aa(ii), atol=1e-5) = }")
    """
    # BatchNormalization returns: gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta
    # BN -> Dense = (gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon)) @ ww + beta @ ww + bb
    # --> gamma / sqrt(self.moving_var + epsilon) @ ww * batch + beta @ ww + bb - gamma * self.moving_mean / sqrt(self.moving_var + epsilon) @ ww
    # --> new_ww = gamma / sqrt(self.moving_var + epsilon) @ ww
    # --> new_bias = beta @ ww + bb - self.moving_mean @ new_ww
    ww = dense_layer.kernel
    bb = dense_layer.bias if dense_layer.use_bias else 0
    batch_std = functional.sqrt(bn_layer.moving_variance + bn_layer.epsilon)

    # ww: [in, out], bb: [out], beta, gamma, moving_mean, moving_variance: [in]
    # -> new_ww: [in, out], new_bias: [out]
    new_ww = ww * (bn_layer.gamma / batch_std)[:, None]
    new_bias = bn_layer.beta[None, :] @ ww + bb - bn_layer.moving_mean[None, :] @ new_ww
    return [new_ww, new_bias[0]]


def fuse_conv_bn_weights(conv_layer, bn_layer):
    """
    Convert model by fusing Conv + batchnorm

    Exampls:
    >>> from keras_cv_attention_models import model_surgery
    >>> mm = keras.applications.ResNet50()
    >>> bb = model_surgery.convert_to_fused_conv_bn_model(mm)
    """
    # BatchNormalization returns: gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta
    # --> new_ww = gamma * conv_w / np.sqrt(var + epsilon)
    # --> new_bb = gamma * (conv_b - mean) / sqrt(var + epsilon) + beta
    batch_std = functional.sqrt(bn_layer.moving_variance + bn_layer.epsilon)

    ww = functional.transpose(conv_layer.depthwise_kernel, [0, 1, 3, 2]) if isinstance(conv_layer, layers.DepthwiseConv2D) else conv_layer.kernel
    bb = conv_layer.bias if conv_layer.use_bias else 0

    new_ww = ww * bn_layer.gamma / batch_std
    new_ww = functional.transpose(new_ww, [0, 1, 3, 2]) if isinstance(conv_layer, layers.DepthwiseConv2D) else new_ww
    new_bb = bn_layer.gamma * (bb - bn_layer.moving_mean) / batch_std + bn_layer.beta
    return [new_ww, new_bb]
    # cc = conv_layer.get_config()
    # cc["use_bias"] = True
    # fused_conv_bn = conv_layer.__class__.from_config(cc)
    # fused_conv_bn.build(conv_layer.input_shape)
    # fused_conv_bn.set_weights([ww, bias])
    # return fused_conv_bn


def convert_to_fused_conv_bn_model(model, verbose=0):
    """
    Convert model by fusing Conv + batchnorm

    Exampls:
    >>> from keras_cv_attention_models import model_surgery
    >>> mm = keras.applications.ResNet50()
    >>> bb = model_surgery.convert_to_fused_conv_bn_model(mm)
    """
    fuse_layer_condition = lambda layer: layer["class_name"] == "BatchNormalization"
    pre_layer_condition = lambda layer: layer.get("class_name") in ["Conv2D", "DepthwiseConv2D"]

    def layer_config_process(pre_layer_config, fused_layer_config):
        pre_layer_config["use_bias"] = True
        return pre_layer_config

    layer_weight_process = lambda pre_fused_layer, fused_layer: fuse_conv_bn_weights(pre_fused_layer, fused_layer)

    return fuse_layer_single_input(
        model=model,
        fuse_layer_condition=fuse_layer_condition,
        pre_layer_condition=pre_layer_condition,
        layer_config_process=layer_config_process,
        layer_weight_process=layer_weight_process,
        verbose=verbose,
    )


def remove_layer_single_input(model, remove_layer_condition, pre_layer_condition=None, layer_config_process=None, layer_weight_process=None, verbose=0):
    return fuse_layer_single_input(model, remove_layer_condition, pre_layer_condition, layer_config_process, layer_weight_process, verbose=verbose)


def fuse_sequential_conv_strict(model, verbose=0):
    """
    Convert model by fusing Conv + batchnorm

    Exampls:
    >>> from keras_cv_attention_models import model_surgery
    >>> mm = keras.applications.ResNet50()
    >>> bb = model_surgery.convert_to_fused_conv_bn_model(mm)
    """
    layer_as_inputs_count = get_layer_as_inputs_count_from_mode_json(model)
    fuse_layer_condition = lambda layer: layer["class_name"] == "Conv2D" and max(layer["config"]["kernel_size"]) == 1
    pre_layer_condition = lambda layer: layer.get("class_name") in ["Conv2D"] and layer_as_inputs_count[layer["name"]] == 1  # pre layer has only 1 output

    def layer_config_process(pre_layer_config, fused_layer_config):
        pre_layer_config["filters"] = fused_layer_config["filters"]
        pre_layer_config["kernel_size"] = pre_layer_config["kernel_size"] if max(pre_layer_config["kernel_size"]) > 1 else fused_layer_config["kernel_size"]
        pre_layer_config["strides"] = pre_layer_config["strides"] if max(pre_layer_config["strides"]) > 1 else fused_layer_config["strides"]
        pre_layer_config["use_bias"] = pre_layer_config["use_bias"] or fused_layer_config["use_bias"]
        pre_layer_config["padding"] = pre_layer_config["padding"] if pre_layer_config["padding"].lower() != "valid" else fused_layer_config["padding"]

        return pre_layer_config

    def layer_weight_process(pre_fused_layer, fused_layer):
        pre_ww = pre_fused_layer.get_weights()
        pre_ww, pre_bias = pre_ww if len(pre_ww) == 2 else (pre_ww, None)
        fused_ww = fused_layer.get_weights()
        fused_ww, fused_bias = fused_ww if len(fused_ww) == 2 else (fused_ww, None)

        fused_ww = np.squeeze(fused_ww)
        new_ww = pre_ww @ fused_ww

        if pre_bias is None and fused_bias is None:
            return [new_ww]
        if pre_bias is None:
            return [new_ww, fused_bias]

        pre_bias = pre_bias @ fused_ww
        new_bias = pre_bias if fused_bias is None else (fused_bias + pre_bias)
        return [new_ww, new_bias]

    return fuse_layer_single_input(
        model=model,
        fuse_layer_condition=fuse_layer_condition,
        pre_layer_condition=pre_layer_condition,
        layer_config_process=layer_config_process,
        layer_weight_process=layer_weight_process,
        verbose=verbose,
    )


def fuse_channel_affine_to_conv_dense(model, verbose=0):
    fuse_layer_condition = lambda layer: layer["class_name"].split(">")[-1] == "ChannelAffine"
    pre_layer_condition = lambda layer: layer.get("class_name") in ["Conv2D", "DepthwiseConv2D", "Dense"]

    def layer_weight_process(pre_fused_layer, fused_layer):
        pre_ww = pre_fused_layer.get_weights()
        pre_ww, pre_bias = pre_ww if len(pre_ww) == 2 else (pre_ww, None)
        fused_ww = fused_layer.get_weights()
        fused_ww, fused_bias = fused_ww if len(fused_ww) == 2 else (fused_ww, None)

        fused_ww = np.squeeze(fused_ww)
        new_ww = pre_ww * (fused_ww[:, None] if pre_ww.shape[-1] == 1 else fused_ww)
        if pre_bias is None and fused_bias is None:
            return [new_ww]
        if pre_bias is None:
            return [new_ww, fused_bias]

        pre_bias = pre_bias * fused_ww
        new_bias = pre_bias if fused_bias is None else (fused_bias + pre_bias)
        return [new_ww, new_bias]

    return fuse_layer_single_input(
        model=model,
        fuse_layer_condition=fuse_layer_condition,
        pre_layer_condition=pre_layer_condition,
        layer_weight_process=layer_weight_process,
        verbose=verbose,
    )


def fuse_reparam_blocks(model, output_layer_key="REPARAM_out", verbose=0):
    # model = convert_to_fused_conv_bn_model(model)
    fuse_layers = []
    fused_reparam_weights = {}  # Record fused weights first, then set_weights after new model built
    for block_out_layer in model.layers:
        if not block_out_layer.name.endswith(output_layer_key) or len(block_out_layer.input) < 2:
            continue

        prefix = block_out_layer.name.split(output_layer_key)[0]
        if verbose > 0:
            print(">>>> block_out_layer.name:", block_out_layer.name, "prefix:", prefix)

        # Just regard the first input as the one with largest kernel_size.
        # It should always with use_bias=True, the bias may comes from fused BatchNorm layer.
        remain_layer = model.get_layer(block_out_layer.input[0].node.layer.name)

        orign_weights = remain_layer.get_weights()
        orign_weight, orign_bias = orign_weights if len(orign_weights) == 2 else (orign_weights[0], 0)
        for input_layer in block_out_layer.input[1:]:
            input_layer = model.get_layer(input_layer.node.layer.name)
            if verbose > 0:
                print("     input_layer.name:", input_layer.name)

            if not input_layer.name.startswith(prefix) or isinstance(input_layer, layers.BatchNormalization):  # Identity branch
                # Conv2D with groups == 1 -> np.eye(filters), DepthWiseConv2D -> 1, COnv2D with groups != 1 -> partial 1
                in_channels, out_channels = remain_layer.weights[0].shape[2], remain_layer.weights[0].shape[3]
                branch_weight = np.zeros([1, 1, in_channels, out_channels], dtype="float32")
                filters, groups = (in_channels, in_channels) if isinstance(remain_layer, layers.DepthwiseConv2D) else (out_channels, remain_layer.groups)
                group_interval = filters // groups
                for ii in range(filters):
                    branch_weight[:, :, ii, ii % group_interval] = 1.0
                branch_bias = 0
                cur_kernel_size = [1] * len(remain_layer.kernel_size)
            elif hasattr(input_layer, "kernel_size"):
                branch_weights = input_layer.get_weights()
                branch_weight, branch_bias = branch_weights if len(branch_weights) == 2 else (branch_weights[0], 0)
                cur_kernel_size = input_layer.kernel_size
                fuse_layers.append(input_layer.name)
                if isinstance(input_layer.input.node.layer, layers.ZeroPadding2D):
                    fuse_layers.append(input_layer.input.node.layer.name)

            if isinstance(input_layer, layers.BatchNormalization):
                if isinstance(remain_layer, layers.DepthwiseConv2D):
                    fake_conv = layers.DepthwiseConv2D(kernel_size=1, use_bias=False)
                else:
                    fake_conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
                _ = fake_conv(initializers.ones()([1, *input_layer.input_shape[1:]]))
                fake_conv.set_weights([branch_weight])  # Previously created identity branch weights
                branch_weight, branch_bias = fuse_conv_bn_weights(fake_conv, input_layer)
                fuse_layers.append(input_layer.name)

            pad = [(remain - cur) // 2 for remain, cur in zip(remain_layer.kernel_size, cur_kernel_size)]  # (3, 3) -> (1, 1)
            branch_weight = functional.pad(branch_weight, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0], [0, 0]])
            orign_weight = (orign_weight - branch_weight) if isinstance(block_out_layer, layers.Subtract) else (orign_weight + branch_weight)
            orign_bias = (orign_bias - branch_bias) if isinstance(block_out_layer, layers.Subtract) else (orign_bias + branch_bias)
        fuse_layers.append(block_out_layer.name)
        fused_reparam_weights[remain_layer.name] = [orign_weight, orign_bias]

    cc = remove_layer_single_input(model, remove_layer_condition=lambda layer: layer["name"] in fuse_layers, verbose=verbose)
    for ii in cc.layers:
        if ii.name in fused_reparam_weights:
            if verbose > 0:
                print(">>>> set_weights layer.name:", ii.name)
            ii.set_weights(fused_reparam_weights[ii.name])
    return cc


def fuse_distill_head(model, head="head", distill_head="distill_head", head_bn="head_bn", distill_head_bn="distill_head_bn"):
    head = model.get_layer(head)
    if head_bn is not None:
        head_bn = model.get_layer(head_bn)
        head_ww, head_bb = fuse_bn_dense_weights(head_bn, head)
    else:
        head_ww, head_bb = head.get_weights()

    if distill_head in model.output_names:
        distill_head = model.get_layer(distill_head)
        if distill_head_bn is not None:
            distill_head_bn = model.get_layer(distill_head_bn)
            distill_ww, distill_bb = fuse_bn_dense_weights(distill_head_bn, distill_head)
        else:
            distill_ww, distill_bb = distill_head.get_weights()
        head_ww, head_bb = (distill_ww + head_ww) / 2, (distill_bb + head_bb) / 2

    if head_bn is not None:
        model = remove_layer_single_input(model, remove_layer_condition=lambda layer: layer["name"] in [head_bn.name])
    model = models.Model(model.inputs, model.get_layer(head.name).output)
    model.get_layer(head.name).set_weights([head_ww, head_bb])
    return model


def convert_layers_to_deploy_inplace(model):
    for ii in model.layers:
        if hasattr(ii, "switch_to_deploy"):
            ii.switch_to_deploy()
    return model


""" TFLite and ONNX """


@backend.register_keras_serializable(package="model_surgery")
class SplitConv2D(layers.Conv2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.super_class = layers.Conv2D

    def build(self, input_shape):
        import tensorflow as tf

        cc = self.get_config().copy()
        cc.update({"groups": 1, "filters": self.filters // self.groups})
        grouped_input_shape = (*input_shape[:-1], input_shape[-1] // self.groups)
        self.convs = []
        for ii in range(self.groups):
            name_scope = self.name + "_{}".format(ii)
            with tf.name_scope(name_scope) as scope:
                cc["name"] = self.name + "_{}".format(ii)
                conv = self.super_class.from_config(cc)
                conv.build(grouped_input_shape)
            self.convs.append(conv)

    def call(self, inputs, **kwargs):
        return functional.concat([conv(ii) for conv, ii in zip(self.convs, functional.split(inputs, self.groups, axis=-1))], axis=-1)


@backend.register_keras_serializable(package="model_surgery")
class SplitScaledStandardizedConv2D(SplitConv2D):
    def __init__(self, gamma=1.0, eps=1e-5, **kwargs):
        from keras_cv_attention_models import attention_layers

        super().__init__(**kwargs)
        self.super_class = attention_layers.ScaledStandardizedConv2D
        self.eps, self.gamma = eps, gamma

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"eps": self.eps, "gamma": self.gamma})
        return base_config


def convert_groups_conv2d_2_split_conv2d(model):
    def __convert_groups_conv2d_2_split_conv2d__(layer):
        if isinstance(layer, layers.Conv2D) and not isinstance(layer, SplitConv2D) and layer.groups != 1:
            aa = layer.get_config()
            # Check if ScaledStandardizedConv2D or typical Conv2D
            bb = SplitScaledStandardizedConv2D.from_config(aa) if hasattr(layer, "gain") else SplitConv2D.from_config(aa)
            # bb.build(layer.input_shape)   # looks like build not working [ ??? ]
            bb(initializers.ones()([1, *layer.input_shape[1:]]))
            wws = functional.split(layer.get_weights()[0], bb.groups, axis=-1)
            if bb.use_bias:
                bbs = functional.split(layer.get_weights()[1], bb.groups, axis=-1)
            if hasattr(layer, "gain"):
                # ScaledStandardizedConv2D with gain from NFNets
                ggs = functional.split(layer.get_weights()[-1], bb.groups, axis=-1)
            for id in range(bb.groups):
                sub_weights = [wws[id].numpy()]
                if bb.use_bias:
                    sub_weights.append(bbs[id].numpy())
                if hasattr(layer, "gain"):
                    sub_weights.append(ggs[id].numpy())
                bb.convs[id].set_weights(sub_weights)
            return bb
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__convert_groups_conv2d_2_split_conv2d__)


def convert_gelu_to_approximate(model, target_activation="gelu/app"):
    from keras_cv_attention_models import common_layers

    def __convert_gelu_to_approximate__(layer):
        if isinstance(layer, layers.Activation) and layer.activation.__name__ == "gelu":
            if target_activation == "gelu/quick":
                return layers.Lambda(common_layers.gelu_quick)
            else:  # "gelu/app"
                return layers.Lambda(lambda xx: functional.gelu(xx, approximate=True))
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__convert_gelu_to_approximate__)


def convert_extract_patches_to_conv(model):
    from keras_cv_attention_models import attention_layers

    def __convert_extract_patches_to_conv__(layer):
        if isinstance(layer, attention_layers.CompatibleExtractPatches):
            aa = layer.get_config()
            aa.update({"force_conv": True})
            bb = attention_layers.CompatibleExtractPatches.from_config(aa)
            bb.build(layer.input_shape)  # No weights for this layer
            return bb
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__convert_extract_patches_to_conv__)


def convert_gelu_and_extract_patches_for_tflite(model):
    print("[Deprecated], use convert_gelu_to_approximate -> convert_extract_patches_to_conv instead")
    model = convert_gelu_to_approximate(model)
    model = convert_extract_patches_to_conv(model)
    return model


def convert_dense_to_conv(model):
    """Convert Dense layers to Conv1D or Conv2D, as TFLite not supporting Dense layers with xnnpack [???]
    >>> from keras_cv_attention_models import model_surgery

    >>> inputs = keras.layers.Input([32, 32, 3])
    >>> nn = keras.layers.Dense(12, use_bias=True)(inputs)
    >>> mm = keras.models.Model(inputs, nn, name='test_dense')
    >>>
    >>> bb = model_surgery.convert_dense_to_conv(mm)
    >>> inputs = tf.random.uniform([1, 32, 32, 3])
    >>> np.allclose(mm(inputs), bb(inputs))
    >>>
    >>> converter = tf.lite.TFLiteConverter.from_keras_model(bb)
    >>> open(mm.name + ".tflite", "wb").write(converter.convert())
    """

    def __convert_dense_to_conv__(layer):
        if isinstance(layer, layers.Dense) and (len(layer.input_shape) == 3 or len(layer.input_shape) == 4):
            target_layer = layers.Conv1D if len(layer.input_shape) == 3 else layers.Conv2D
            bb = target_layer(layer.units, kernel_size=1, use_bias=layer.use_bias, name=layer.name + "conv")
            bb(initializers.ones()([1, *layer.input_shape[1:]]))

            wws = [np.reshape(ii, jj.shape) for ii, jj in zip(layer.get_weights(), bb.get_weights())]
            bb.set_weights(wws)
            return bb
        return layer

    input_tensors = layers.Input(model.input_shape[1:])
    return models.clone_model(model, input_tensors=input_tensors, clone_function=__convert_dense_to_conv__)


def convert_to_fixed_batch_size(model, batch_size=1):
    """Fixing batch_size avoiding dynamic tensors, for supporting TFLite xnnpack. https://github.com/tensorflow/tensorflow/issues/60762"""
    input_tensors = [layers.Input(ii.shape[1:], dtype=ii.dtype, name=ii.name, batch_size=batch_size) for ii in model.inputs]
    return models.clone_model(model, input_tensors=input_tensors)


def prepare_for_tflite(model, batch_size=1):
    model = convert_to_fixed_batch_size(model) if batch_size > 0 else convert_dense_to_conv(model)
    # model = convert_groups_conv2d_2_split_conv2d(model)
    # model = convert_gelu_to_approximate(model)
    model = convert_extract_patches_to_conv(model)
    return model


def _tf_export_onnx_(model, filepath=None, fuse_conv_bn=True, batch_size=None, simplify=False, **kwargs):
    import tensorflow as tf
    import tf2onnx

    model = convert_extract_patches_to_conv(model)  # ExtractImagePatches not supported in ONNX
    model = convert_groups_conv2d_2_split_conv2d(model)  # StatefulPartitionCall not supported in ONNX
    if fuse_conv_bn:
        print("Fuse Conv + BN")
        model = convert_to_fused_conv_bn_model(model)

    spec = (tf.TensorSpec((batch_size, *model.input_shape[1:]), tf.float32, name="input"),)
    filepath = filepath or model.name + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=filepath, **kwargs)
    print("Exported onnx:", filepath)

    if simplify:
        import onnx, onnxsim

        print("Running onnxsim.simplify...")
        model_proto, check = onnxsim.simplify(model_proto)
        if check:
            with open(filepath, "wb") as ff:
                ff.write(model_proto.SerializeToString())
            print("Exported simplified onnx:", filepath)
        else:
            print("[Error] failed to simplify onnx:", filepath)


def export_onnx(model, filepath=None, fuse_conv_bn=True, batch_size=None, simplify=False, **kwargs):
    """
    >>> # !pip install onnx tf2onnx onnxsim onnxruntime
    >>> from keras_cv_attention_models import volo, nat, model_surgery
    >>> mm = nat.DiNAT_Small(pretrained=True)
    >>> model_surgery.export_onnx(mm, fuse_conv_bn=True, batch_size=1, simplify=True)
    >>> # Exported simplified onnx: dinat_small.onnx

    # Run test
    >>> from keras_cv_attention_models.imagenet import eval_func
    >>> aa = eval_func.ONNXModelInterf(mm.name + '.onnx')
    >>> inputs = np.random.uniform(size=[1, *mm.input_shape[1:]]).astype('float32')
    >>> print(f"{np.allclose(aa(inputs), mm(inputs), atol=1e-5) = }")
    >>> # np.allclose(aa(inputs), mm(inputs), atol=1e-5) = True
    """
    if hasattr(model, "export_onnx"):
        model.export_onnx(filepath=filepath, simplify=simplify, **kwargs)
    else:
        _tf_export_onnx_(model, filepath=filepath, fuse_conv_bn=fuse_conv_bn, batch_size=batch_size, simplify=simplify, **kwargs)


""" Convert previous weights to new version """


def swin_convert_pos_emb_mlp_to_MlpPairwisePositionalEmbedding_weights(source_file, save_path):
    """
    Convert previous kecam <= 1.3.18 Swinv2 weights using PairWiseRelativePositionalEmbedding -> mlp_block
    to MlpPairwisePositionalEmbedding layer.

    Example:
    >>> from keras_cv_attention_models.model_surgery import swin_convert_pos_emb_mlp_to_MlpPairwisePositionalEmbedding_weights
    >>> source_file = os.path.expanduser("~/.keras/models/swin_transformer_v2_tiny_window16_256_imagenet.h5")
    >>> target_file = "swin_transformer_v2_tiny_window16_256_imagenet_modified.h5"
    >>> swin_convert_pos_emb_mlp_to_MlpPairwisePositionalEmbedding_weights(source_file, target_file)
    >>> # Saved to: swin_transformer_v2_tiny_window16_256_imagenet_modified.h5
    # Test
    >>> from keras_cv_attention_models import test_images, swin_transformer_v2
    >>> mm = swin_transformer_v2.SwinTransformerV2Tiny_window16(pretrained=target_file)
    >>> mm.decode_predictions(mm(mm.preprocess_input(test_images.cat())))
    """
    from keras_cv_attention_models import download_and_load

    with download_and_load.H5orKerasFileReader(source_file) as ff:
        bb = {kk: np.array(vv) for kk, vv in ff.items()}
    layer_names = list(bb.keys())

    pos_emb_weight_names = ["hidden_weight:0", "hidden_bias:0", "out:0"]
    for layer_name in layer_names:
        if layer_name.endswith("_meta_dense_1"):
            dense_1_name, dense_2_name, pos_layer_name = layer_name, layer_name.replace("meta_dense_1", "meta_dense_2")
            pos_layer_name = layer_name.replace("meta_dense_1", "pos_emb")
            if dense_1_name not in bb or dense_2_name not in bb:
                print("[WARNING] cannot match {}, requires source containing {} and {}".format(pos_layer_name, dense_1_name, dense_2_name))
                continue

            print("Layer:", pos_layer_name)
            dense_1 = bb.pop(dense_1_name)
            dense_2 = bb.pop(dense_2_name)
            weight_names = [pos_layer_name + "/" + ii for ii in pos_emb_weight_names]
            print("    weight_names:", weight_names, "weight shapes:", [ii.shape for ii in dense_1 + dense_2])
            bb[pos_layer_name] = dict(zip(weight_names, dense_1 + dense_2))
    download_and_load.save_weights_to_hdf5_file(save_path, bb)
    print(">>>> Saved to:", save_path)
