import time, threading, queue, h5py, argparse
import numpy as np
import pvaccess as pva

class daqSimuEPICS:
    def __init__(self, h5, daq_freq):
        self.arraySize = None
        
        if h5 is None:
            self.frames = np.random.randint(0, 256, size=(1000, 2048, 2048), dtype=np.uint16)
        else:
            with h5py.File(h5, 'r') as h5fd:
                self.frames = h5fd['frames'][:]

        self.rows, self.cols = self.frames.shape[-2:]

        self.daq_freq = daq_freq
        # make data acq and streaming async so as to overlap them for more accurate daq freq
        self.tq = queue.Queue()
        threading.Thread(target=self.frame_publisher, daemon=True).start()

        self.server = pva.PvaServer()
        self.server.start()

        self.first_frame = True

        self.channel = 'pvapy:image'

    def frame_publisher(self, extraFieldsPvObject=None):
        while True:
            frm_id = self.tq.get()
            self.tq.task_done()

            if extraFieldsPvObject is None:
                nda = pva.NtNdArray()
            else:
                nda = pva.NtNdArray(extraFieldsPvObject.getStructureDict())

            nda['uniqueId'] = frm_id
            nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(14))
            dims = [pva.PvDimension(self.rows, 0, self.rows, 1, False), \
                    pva.PvDimension(self.cols, 0, self.cols, 1, False)]
            nda['dimension'] = dims
            nda['descriptor'] = 'PvaPy Simulated Image'
            nda['value'] = {'ushortValue': self.frames[frm_id].flatten()}
            if extraFieldsPvObject is not None:
                nda.set(extraFieldsPvObject)

            # print("sending  frame %d @ %f" % (frm_id, time.time()))
            if self.first_frame:
                self.server.addRecord(self.channel, nda)
                self.first_frame = False
                # at the very begining the channel is not there
                # by the time epics established connection, you may already replaced image
                # depends on how muchtime it takes to establish connection and FPS
                # wait for 0.5 is not a neat solution though
                time.sleep(0.5) # make sure it's propogated
            else:
                self.server.update(self.channel, nda)

            print("sent     frame %d @ %f" % (frm_id, time.time()))

    def start(self, ):
        for fid in range(self.frames.shape[0]):
            time.sleep(1 / self.daq_freq)
            self.tq.put(fid)
            print("produced frame %d @ %f" % (fid, time.time()))
        self.tq.join()

        # there should be a better way to make sure all frames are ack'ed before exiting
        time.sleep(5) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data streaming from detector using EPICS')
    parser.add_argument('-ifn', type=str,   default=None, help='h5 file to be streamed')
    parser.add_argument('-fps', type=float, default=1,    help='frame per second')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    daq = daqSimuEPICS(h5=args.ifn, daq_freq=args.fps)

    daq.start()
