import h5py, logging
from multiprocessing import Process, Queue
import numpy as np

class asyncHDFWriter:
    def __init__(self, fname, compression=False):
        self.fname = fname
        self.h5fd = None
        self.buffer = {}
        self.task_q = Queue(maxsize=-1)
        self.compression = compression
    '''
    Args:
        ddict: dict of datasets to be written to h5, data will be concatenated on
               the first dimension
    '''
    def append2write(self, ddict):
        self.task_q.put(ddict)

    def start(self,):
        p = Process(target=self.write2file)
        p.start()
        logging.info(f"Async writer to {self.fname} started ...")

    def write2file(self,):
        while True:
            try:
                ddict = self.task_q.get()
            except queue.Empty:
                continue
            except:
                logging.error("Something else of the writer Queue went wrong")
                continue

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
