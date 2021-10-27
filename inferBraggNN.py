
from BraggNN import BraggNN
from trtUtil import engine_build_from_onnx, mem_allocation, inference
import logging, time, threading
import numpy as np

class inferBraggNN:
    def __init__(self, mbsz, onnx_mdl, patch_tq):
        self.patch_tq = patch_tq
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl

    def start(self, ):
        threading.Thread(target=self.batch_infer, daemon=True).start()

    def batch_infer(self, ):
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()
        logging.info("[%.3f] Inference engine initialization completed!" % (time.time(), ))
        while True:
            in_mb = []
            for i in range(self.mbsz):
                _p = self.patch_tq.get()
                in_mb.append(_p)
                self.patch_tq.task_done()
            tick = time.time()
            in_mb = np.array(in_mb)
            np.copyto(self.trt_hin, in_mb.ravel())
            pred = inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream)
            t_e2e = 1000 * (time.time() - tick)
            logging.info("[%.3f] a batch of %d patches infered in %.3f ms, %d patches pending infer. %.4f, %.4f" % (\
                         time.time(), self.mbsz, t_e2e, self.patch_tq.qsize(), pred[0], pred[1]))
