from BraggNN import BraggNN
from bm_tensorRT_infer import allocate_buffers, build_engine_onnx, do_inference

class inferBraggNN:
    def __init__(self, mbsz, psz, mdl):
        self.trt_engine = build_engine_onnx(mdl)
        inputs, outputs, bindings, stream = allocate_buffers(self.trt_engine)
