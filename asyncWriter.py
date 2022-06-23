import h5py, logging, zmq, queue, threading
from multiprocessing import Process, Queue
import numpy as np

class asyncHDFWriter(threading.Thread):
    def __init__(self, fname, compression=False):
        threading.Thread.__init__(self)
        self.daemon = True
        self.fname = fname
        self.h5fd = None
        self.task_q = Queue(maxsize=-1)
        self.compression = compression
    '''
    Args:
        ddict: dict of datasets to be written to h5, data will be concatenated on
               the first dimension
    '''
    def append2write(self, ddict):
        self.task_q.put(ddict)

    def run(self,):
        logging.info(f"Async writer to {self.fname} started ...")
        while True:
            ddict = self.task_q.get()
            if self.h5fd is None:
                self.h5fd = h5py.File(self.fname, 'w')
                for key, data in ddict.items():
                    dshape = list(data.shape)
                    dshape[0] = None
                    if self.compression:
                        self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape, compression="gzip")
                    else:
                        self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape)
                    logging.info(f"{data.shape} samples added to '{key}' of {self.fname}")
            else:
                for key, data in ddict.items():
                    self.h5fd[key].resize((self.h5fd[key].shape[0] + data.shape[0]), axis=0)
                    self.h5fd[key][-data.shape[0]:] = data
                    logging.info(f"{data.shape} samples added to '{key}' of {self.fname}, now has {self.h5fd[key].shape}")
            self.h5fd.flush()

'''
as zmq socket is not pickable, Thread, instead of Process should be used
or move the socket creation to the sending function before loop
'''
class asyncZMQWriter(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.daemon = True
        self.port = port
        self.task_q = queue.Queue(maxsize=-1)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{self.port}")

    def append2write(self, ddict):
        self.task_q.put(ddict)

    def run(self,):
        logging.info(f"Async writer to ZMQ:{self.port} started ...")
        while True:
            ddict = self.task_q.get()
            ret = self.publisher.send_pyobj(ddict)
            logging.info(f"datasets {ddict.keys()} have been published via ZMQ {ret}")