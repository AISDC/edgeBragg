import pycuda.driver as cuda
import tensorrt as trt

def engine_build_from_onnx(onnx_mdl):
    builder = trt.Builder(TRT_LOGGER)
    config  = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.max_workspace_size = 256 * (1 << 20) # the maximum size that any layer in the network can use

    network = builder.create_network(EXPLICIT_BATCH)
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    success = parser.parse_from_file(onnx_mdl)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return None

    return builder.build_engine(network, config)

def mem_allocation(context):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input  = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)

    # Allocate device memory for inputs and outputs.
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input, h_output, d_input, d_output, stream

def inference(context, h_input, h_output, d_input, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference.
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize the stream
    stream.synchronize()

    # Return the host
    return h_output

engine = engine_build_from_onnx(onnx_mdl)
context = engine.create_execution_context()
h_input, h_output, d_input, d_output, stream = mem_allocation(context)

pred = inference(context, h_input, h_output, d_input, d_output, stream)

