import logging
from BraggNN import scriptpth2onnx
from inferBraggNN import inferBraggNNtrt, inferBraggNNTorch
from multiprocessing import Queue

class pvaClient:
    def __init__(self, mbsz, psz, trt, pth, ofname):
        self.psz = psz
        self.patch_tq = Queue(maxsize=-1)
        if trt:
            onnx_fn = scriptpth2onnx(pth, mbsz, psz=psz)
            self.infer_engine = inferBraggNNtrt(mbsz=mbsz, onnx_mdl=onnx_fn, patch_tq=self.patch_tq, \
                                                ofname=ofname)
        else:
            self.infer_engine = inferBraggNNTorch(script_pth=pth, patch_tq=self.patch_tq, \
                                                  ofname=ofname)
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = None
        self.frame_tq = Queue(maxsize=-1)
        self.infer_engine.start()
        
    def monitor(self, pv):
        uid = pv['uniqueId']

        # ignore the 1st empty frame when use sv simulator
        if self.recv_frames is None:
            self.recv_frames = 0
            return 

        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1
        
        # problem to pickle PvObject, so just unpack and push to queue
        frm_id = pv['uniqueId']
        dims  = pv['dimension']
        rows  = dims[0]['size']
        cols  = dims[1]['size']
        codec = pv["codec"]
        if len(codec['name']) > 0:
            data_codec = pv['value'][0]['ubyteValue']
            compressed = pv["compressedSize"]
            uncompressed = pv["uncompressedSize"]
        else:
            compressed   = None
            uncompressed = None
            data_codec   = pv['value'][0]['ushortValue']

        self.frame_tq.put((frm_id, data_codec, compressed, uncompressed, codec, rows, cols))
        logging.info("received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     uid, self.recv_frames, uid - self.base_seq_id + 1, self.frame_tq.qsize()))