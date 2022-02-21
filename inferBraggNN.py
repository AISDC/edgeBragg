import logging, time, threading, torch
import numpy as np

class inferBraggNNtrt:
    def __init__(self, mbsz, onnx_mdl, patch_tq):
        self.patch_tq = patch_tq
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl

    def start(self, ):
        threading.Thread(target=self.batch_infer, daemon=True).start()

    def batch_infer(self, ):
        from trtUtil import engine_build_from_onnx, mem_allocation, inference
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()
        logging.info("[%.3f] TensorRT Inference engine initialization completed!" % (time.time(), ))

        while True:
            batch_tick = time.time()
            in_mb = self.patch_tq.get()

            comp_tick = time.time()
            np.copyto(self.trt_hin, in_mb.ravel())
            pred = inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream)
            t_comp  = 1000 * (time.time() - comp_tick)
            t_batch = 1000 * (time.time() - batch_tick)
            logging.info("[%.3f] a batch of %d patches was infered in %.3f ms (computing: %.3f ms), %d batches pending infer." % (\
                         time.time(), self.mbsz, t_batch, t_comp, self.patch_tq.qsize()))

class inferBraggNNTorch:
    def __init__(self, script_pth, patch_tq):
        self.patch_tq = patch_tq
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.BraggNN = torch.jit.load(script_pth, map_location='cuda:0')
        else:
            self.BraggNN = torch.jit.load(script_pth, map_location='cpu')
        self.psz = self.BraggNN.input_psz.item()
        logging.info("[%.3f] Inference engine initialization completed!" % (time.time(), ))

    def start(self, ):
        threading.Thread(target=self.batch_infer, daemon=True).start()

    def batch_infer(self, ):
        while True:
            batch_tick = time.time()
            in_mb = self.patch_tq.get()
            input_tensor = torch.from_numpy(in_mb)
            comp_tick = time.time()
            with torch.no_grad():
                pred = self.BraggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
            t_comp  = 1000 * (time.time() - comp_tick)
            t_batch = 1000 * (time.time() - batch_tick)
            logging.info("[%.3f] a batch of %d patches infered in %.3f ms (computing: %.3f ms), %d batches pending infer." % (\
                         time.time(), in_mb.shape[0], t_batch, t_comp, self.patch_tq.qsize()))
