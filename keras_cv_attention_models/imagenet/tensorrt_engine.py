import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

try:
    # Use autoprimaryctx if available (pycuda >= 2021.1) to prevent issues with other modules that rely on the primary device context.
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def init_mean_std_by_rescale_mode(rescale_mode):
    if isinstance(rescale_mode, (list, tuple)):  # Specific mean and std
        mean, std = rescale_mode
    elif rescale_mode == "torch":
        mean = np.array([0.485, 0.456, 0.406]).astype("float32") * 255.0
        std = np.array([0.229, 0.224, 0.225]).astype("float32") * 255.0
    elif rescale_mode == "tf":  # [0, 255] -> [-1, 1]
        mean, std = 127.5, 127.5
        # mean, std = 127.5, 128.0
    elif rescale_mode == "tf128":  # [0, 255] -> [-1, 1]
        mean, std = 128.0, 128.0
    elif rescale_mode == "raw01":
        mean, std = 0, 255.0  # [0, 255] -> [0, 1]
    else:
        mean, std = 0, 1  # raw inputs [0, 255]
    return mean, std


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, cache_file, rescale_mode="torch"):
        super().__init__()
        self.data, self.cache_file, self.rescale_mode = data, cache_file, rescale_mode
        self.built = False

    def build(self, target_shape, batch_size, data_format="channels_last"):
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        mean, std = init_mean_std_by_rescale_mode(self.rescale_mode)
        target_shape = target_shape if isinstance(target_shape, (list, tuple)) else [target_shape, target_shape]
        data = self.data
        if isinstance(data, str):
            from PIL import Image

            print(">>>> Read input data from path:", data)
            target_shape = list(target_shape)[::-1]  # [width, height] for Image.resize
            data = [np.array(Image.open(os.path.join(data, ii)).resize(target_shape)) for ii in os.listdir(data)]
        else:
            resize_shape = list(target_shape)[::-1]  # [width, height] for Image.resize
            target_shape = tuple(target_shape)
            data = [ii if tuple(ii.shape[:2]) == target_shape else np.array(Image.fromarray(ii).resize(resize_shape)) for ii in data]
        data = (np.array(data) - mean) / std
        if data_format != "channels_last":
            data = data.transpose([0, 3, 1, 2])
        self.built_data, self.data_format = data, data_format

        self.batch_size, self.target_shape = batch_size, target_shape
        self.current_index = 0
        self.device_input = cuda.mem_alloc(data[0].nbytes * self.batch_size)  # Allocate enough memory for a whole batch.
        self.built = True
        print(">>>> Built data info: data.shape={}, data.min={:.4f}, data.max={:.4f}".format(data.shape, data.min(), data.max()))

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.built_data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print(">>>> Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.built_data[self.current_index : self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_onnx_engine_one_input(model_file, engine_path=None, int8_calibrator=None, batch_size=-1, data_format="auto", max_workspace_size=-1):
    # model_file, engine_path, int8_calibrator, batch_size, data_format, max_workspace_size = "aaa.onnx", None, None, -1, "auto", -1
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    max_workspace_size = max_workspace_size if max_workspace_size > 0 else 8
    # config.max_workspace_size = max_workspace_size * (2 ** 30)  # 8 GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, pool_size=max_workspace_size * (2**30))

    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(model_file, "rb") as model:
        parser.parse(model.read())
    assert parser.num_errors == 0

    inputs = network.get_input(0)
    input_shape = inputs.shape
    if batch_size > 0:
        input_shape = [batch_size] + list(input_shape[1:])
        network.get_input(0).shape = input_shape
    else:
        batch_size = inputs.shape[0]
    # builder.max_batch_size = batch_size
    print(">>>> Input name: {}, shape: {}, dtype: {}".format(inputs.name, input_shape, inputs.dtype))

    outputs = [network.get_output(ii) for ii in range(network.num_outputs)]
    for output in outputs:
        print(">>>> Output name: {}, shape: {}, dtype: {}".format(output.name, output.shape, output.dtype))

    if int8_calibrator is None:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.INT8)
        if data_format == "auto":
            data_format = "channels_last" if input_shape[-1] < input_shape[1] else "channels_first"
        data_input_shape = input_shape[1:-1] if data_format == "channels_last" else input_shape[2:]
        print("data_format = {}, data_input_shape = {}".format(data_format, data_input_shape))
        int8_calibrator.build(data_input_shape, batch_size, data_format=data_format)
        config.int8_calibrator = int8_calibrator

    # Dynamic batch_size
    # profile = builder.create_optimization_profile()
    # input_shape = network.get_input(0).shape[1:]
    # profile.set_shape(network.get_input(0).name, (1, *input_shape), (1, *input_shape), (builder.max_batch_size, *input_shape))
    # config.add_optimization_profile(profile)

    # engine = builder.build_engine(network, config)
    engine = builder.build_serialized_network(network, config)
    if engine_path is None:
        engine_path = "{}_{}.trt".format(os.path.splitext(model_file)[0], "float16" if int8_calibrator is None else "int8")
    with open(engine_path, "wb") as ff:
        print(">>>> Serializing engine to file: {:}".format(engine_path))
        ff.write(engine)
    return engine


class EngineInferenceOneInOneOut:
    """
    >>> !pip install tensorrt pycuda
    >>> import torch
    >>> from keras_cv_attention_models.imagenet import tensorrt_engine
    >>> aa = tensorrt_engine.ImageCalibrator('calibration_imagenet/', 'foo.cache')
    >>> ee = tensorrt_engine.build_onnx_engine_one_input('aaa.onnx', int8_calibrator=aa)
    >>> cc = tensorrt_engine.EngineInferenceOneInOneOut(ee, max_batch_size=4)
    >>> print(cc(np.ones([1, 3, 224, 224])).shape)
    >>> # (1, 1000)
    >>> print(cc(np.ones([4, 3, 224, 224])).shape)
    >>> # (4, 1000)
    """

    def __init__(self, engine, max_batch_size=1):
        if isinstance(engine, str):
            with open(engine_path, "rb") as ff:
                engine = ff.read()
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(engine)
        assert runtime

        input_binding, output_binding = None, None
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                input_binding = binding
            else:
                output_binding = binding
        assert input_binding
        assert output_binding

        self.input_shape = [max_batch_size, *engine.get_tensor_shape(input_binding)[1:]]
        self.input_dtype = trt.nptype(engine.get_tensor_dtype(input_binding))
        size = trt.volume(engine.get_tensor_shape(input_binding)) * max_batch_size
        self.host_input = cuda.pagelocked_empty(shape=[size], dtype=self.input_dtype)
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)

        self.output_shape = [max_batch_size, *engine.get_tensor_shape(output_binding)[1:]]
        self.output_dtype = trt.nptype(engine.get_tensor_dtype(output_binding))
        size = trt.volume(engine.get_tensor_shape(output_binding)) * max_batch_size
        self.host_output = cuda.pagelocked_empty(shape=[size], dtype=self.output_dtype)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)

        self.output_dim = self.output_shape[1:]
        self.output_ravel_dim = self.host_output.shape[0] // max_batch_size
        self.allocations = [int(self.cuda_input), int(self.cuda_output)]
        self.max_batch_size = max_batch_size

        self.engine = engine
        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()

    def __call__(self, imgs):
        batch_size = imgs.shape[0]
        if batch_size > self.max_batch_size:
            print(f"Warning: provided input with batch_size={batch_size} exceeds max_batch_size={self.max_batch_size}")
            batch_size = self.max_batch_size
            imgs = imgs[: self.max_batch_size]

        inputs = imgs.ravel()
        # self.context.set_binding_shape(0, imgs.shape)
        np.copyto(self.host_input[: inputs.shape[0]], imgs.ravel())
        cuda.memcpy_htod_async(self.cuda_input, self.host_input[: inputs.shape[0]], self.stream)
        # Run inference asynchronously, same function in cpp is `IExecutionContext::enqueueV2`
        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_output[: batch_size * self.output_ravel_dim], self.cuda_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        return self.host_output[: batch_size * self.output_ravel_dim].reshape([batch_size, *self.output_dim]).copy()


if __name__ == "__main__":
    import torch
    import torchvision

    mm = torchvision.models.resnet50(pretrained=True)
    torch.onnx.export(mm, torch.ones([1, 3, 224, 224]), "aaa.onnx")
    aa = ImageCalibrator("calibration_imagenet/", "foo.cache")
    ee = build_onnx_engine_one_input("aaa.onnx", int8_calibrator=aa)
    cc = EngineInferenceOneInOneOut(ee)
    print(cc(np.ones([1, 3, 224, 224])).shape)
