#! /homes/zhengchun.liu/usr/miniconda3/envs/trt/bin/python 

import numpy as np
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import time, argparse

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
    # context.execute_v2(bindings=bindings, )

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
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None

    config  = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.max_workspace_size = 1 * (1 << 30)
    return builder.build_engine(network, config)

def pth2onnx(args):
    from BraggNN import BraggNN
    import torch 
    model = BraggNN(imgsz=args.psz, fcsz=(16, 8, 4, 2))

    model.load_state_dict(torch.load(args.mdl, map_location=torch.device('cpu')))
    # model = model.cuda()
    dummy_input = torch.randn(args.mbsz, 1, args.psz, args.psz, dtype=torch.float32, device='cpu')

    input_names  = ('patch', )
    output_names = ('ploc',  )

    onnx_fn = args.mdl.replace(".pth", ".onnx")
    torch.onnx.export(model, dummy_input, onnx_fn, verbose=False, \
                      input_names=input_names, output_names=output_names)
    return onnx_fn

def main():
    parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
    parser.add_argument('-gpus',   type=str, default="0", help='the GPU to use')
    parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
    parser.add_argument('-psz',    type=int, default=15, help='input size')
    parser.add_argument('-samples',type=int, default=10240, help='sample size')
    parser.add_argument('-warmup', type=int, default=20, help='warm up batches')
    parser.add_argument('-ifn',    type=str, default=None, help='input h5 file')
    parser.add_argument('-mdl',    type=str, default='models/fc16_8_4_2-sz15.onnx', help='model weights')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if '.pth' in args.mdl:
        onnx_mdl = pth2onnx(args)
    else:
        onnx_mdl = args.mdl
    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_mdl)

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Contexts are used to perform inference.
    batch_time = []
    context = engine.create_execution_context()
    for i in range(args.warmup + args.samples // args.mbsz):
        patches = np.random.rand(args.mbsz, 1, args.psz, args.psz).astype(np.float32).ravel()
        np.copyto(inputs[0].host, patches)

        tick = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        t_e2e = 1000 * (time.time() - tick)
        if i >= args.warmup:
            batch_time.append(t_e2e)

        # print("batch %d takes %.3f ms (%.3f ms / sample)" % (i, t_e2e, t_e2e/args.mbsz))
    
    print("[TRT] BS=%d, batches=%d, psz=%d; time per batch: min: %.3f ms, median: %.3f ms, max: %.3f ms; rate: %.2f us/sample" % (\
          args.mbsz, len(batch_time), args.psz, np.min(batch_time), np.median(batch_time), np.max(batch_time), \
          1000 * np.median(batch_time) / args.mbsz))

if __name__ == '__main__':
    main()
