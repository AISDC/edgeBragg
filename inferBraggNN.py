from BraggNN import BraggNN
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
        logging.info("[%.3f] Inference engine initialization completed!" % (time.time(), ))

        while True:
            batch_tick = time.time()
            in_mb = self.patch_tq.get()

            comp_tick = time.time()
            np.copyto(self.trt_hin, in_mb.ravel())
            pred = inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream)
            t_comp  = 1000 * (time.time() - comp_tick)
            t_batch = 1000 * (time.time() - batch_tick)
            logging.info("[%.3f] a batch of %d patches infered in %.3f ms (computing: %.3f ms), %d batches pending infer. %.4f, %.4f" % (\
                         time.time(), self.mbsz, t_batch, t_comp, self.patch_tq.qsize(), pred[0], pred[1]))

class inferBraggNNTorch:
    def __init__(self, pth_mdl, psz, patch_tq, fcsz=(16, 8, 4, 2)):
        self.psz = psz
        self.patch_tq = patch_tq
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BraggNN  = BraggNN(imgsz=psz, fcsz=fcsz) # should use the same argu as it in the training.
        self.BraggNN .load_state_dict(torch.load(pth_mdl, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            self.BraggNN = self.BraggNN.to(self.torch_dev)
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
