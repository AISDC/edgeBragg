import h5py, logging
from multiprocessing import Process, Queue
import numpy as np

class resWriter:
    def __init__(self, fname, batch_sz):
        self.fname = fname
        self.batch_sz = batch_sz
        self.batch_no = 1

        self.ploc_buf = None
        self.patch_buf= None

        self.task_q   = Queue(maxsize=-1)

    def append2write(self, ploc, patch):
        assert ploc.shape[0] == patch.shape[0], "size of data entries do not match"
        self.task_q.put((ploc, patch))

    def start(self,):
        p = Process(target=self.peak_info_writing)
        p.start()
        logging.info("Async writer started ...")

    def peak_info_writing(self,):
        while True:
            try:
                ploc, patch = self.task_q.get()
            except queue.Empty:
                continue
            except:
                logging.error("Something else of the writer Queue went wrong")
                continue
            if self.ploc_buf is None:
                self.ploc_buf = ploc
                self.patch_buf= patch
            else:
                self.ploc_buf = np.concatenate([self.ploc_buf,  ploc],  axis=0)
                self.patch_buf= np.concatenate([self.patch_buf, patch], axis=0)

            bufsz = self.ploc_buf.shape[0]
            logging.info(f"{ploc.shape[0]} peaks added to buffer, {bufsz} toward {self.batch_sz} pending write.")

            if bufsz >= self.batch_sz:
                ofname = f"{self.fname}.{self.batch_no}"
                with h5py.File(ofname, 'w') as fd:
                    fd.create_dataset('ploc',  data=self.ploc_buf[:self.batch_sz])
                    fd.create_dataset('patch', data=self.patch_buf[:self.batch_sz])
                self.ploc_buf = self.ploc_buf[self.batch_sz:]
                self.patch_buf= self.patch_buf[self.batch_sz:]
                self.batch_no += 1
                logging.info(f"{self.batch_sz} data points have been written to {ofname}.")
