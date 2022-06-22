import logging
from multiprocessing import Queue

class pvaClient:
    def __init__(self, tq_frame, dtype):
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = 0
        self.tq_frame = tq_frame
        self.dtype = dtype

    # this function will be triggered to call by pva when there is a new frame
    def monitor(self, pv):
        uid = pv['uniqueId']

        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1

        frm_id= pv['uniqueId']
        dims  = pv['dimension']
        rows  = dims[0]['size']
        cols  = dims[1]['size']
        codec = pv["codec"]
        if len(codec['name']) > 0:
            data_codec   = pv['value'][0]['ubyteValue']
            compressed   = pv["compressedSize"]
            uncompressed = pv["uncompressedSize"]
        else:
            compressed   = None
            uncompressed = None
            data_codec   = pv['value'][0][self.dtype]

        self.tq_frame.put((frm_id, data_codec, compressed, uncompressed, codec, rows, cols))
        logging.info("received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     uid, self.recv_frames, uid - self.base_seq_id + 1, self.tq_frame.qsize()))