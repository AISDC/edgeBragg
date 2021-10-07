import time, threading, queue
import numpy as np
import pvaccess as pva

class daqSimuEPICS:
    def __init__(self, h5, daq_freq):
        self.arraySize = None
        
        self.frames = np.random.randint(0, 256, size=(1000, 128, 256), dtype=np.uint16)
        self.rows, self.cols = self.frames.shape[-2:]

        self.daq_freq = daq_freq
        self.tq = queue.Queue()
        threading.Thread(target=self.frame_publisher, daemon=True).start()

        self.server = pva.PvaServer()
        self.server.start()

        self.first_frame = True

        self.channel = 'pvapy:image'

    def frame_publisher(self, extraFieldsPvObject=None):
        while True:
            frm_id = self.tq.get()
            print("sending frame %d @ %f" % (frm_id, time.time()))

            if extraFieldsPvObject is None:
                nda = pva.NtNdArray()
            else:
                nda = pva.NtNdArray(extraFieldsPvObject.getStructureDict())
                
            nda['uniqueId'] = frm_id
            nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(14))
            dims = [pva.PvDimension(self.cols, 0, self.cols, 1, False), \
                    pva.PvDimension(self.rows, 0, self.rows, 1, False)]
            nda['dimension'] = dims
            nda['descriptor'] = 'PvaPy Simulated Image'
            nda['attribute'] = [pva.NtAttribute('rows', pva.PvInt(self.rows)), \
                                pva.NtAttribute('cols', pva.PvInt(self.cols))]

            nda['value'] = {'ushortValue': self.frames[frm_id].flatten()}
            # nda.set(extraFieldsPvObject)

            if self.first_frame:
                self.server.addRecord(self.channel, nda)
                self.first_frame = False
            else:
                self.server.update(self.channel, nda)

            self.tq.task_done()

    def start(self, ):
        for fid in range(self.frames.shape[0]):
            time.sleep(1 / self.daq_freq)
            self.tq.put(fid)
            print("produced frame %d @ %f" % (fid, time.time()))
        self.tq.join()


if __name__ == '__main__':
    daq = daqSimuEPICS(h5=None, daq_freq=1)

    daq.start()