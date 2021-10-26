#! /homes/zhengchun.liu/usr/miniconda3/envs/trt/bin/python 

import numpy as np
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import time, torch

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size  = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config  = builder.create_builder_config()
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = 1 * (1 << 30)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    return builder.build_engine(network, config)

def main():
    onnx_model_file = 'models/fc16_8_4_2-sz15.onnx'

    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    mbsz = 512
    batch_latency = []
    for i in range(10):
        patches = np.random.rand(mbsz, 1, 15, 15).astype(np.float32).ravel()
        tick = time.time()
        np.copyto(inputs[0].host, patches)
        # Contexts are used to perform inference.
        context = engine.create_execution_context()

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # print(trt_outputs[0].shape)

        t_e2e = 1000 * (time.time() - tick)
        batch_latency.append(t_e2e)
        print("batch %d takes %.3f ms (%.3f ms / sample)" % (i, t_e2e, t_e2e/mbsz))

if __name__ == '__main__':
    main()