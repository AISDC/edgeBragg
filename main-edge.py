
import time, queue, threading
import torch
from pvaccess import Channel
from pvaccess import PvObject
import numpy as np 

from BraggNN import BraggNN
from preprocess import frame_peak_patches_gcenter as frame2patch

class pvaClient:
    def __init__(self, ):
        self.psz = 15
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BraggNN  = BraggNN(imgsz=self.psz, fcsz=(16, 8, 4, 2)) # should use the same argu as it in the training.
        mdl_fn = 'models/fc16_8_4_2-sz15.pth'
        self.BraggNN .load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            self.BraggNN = self.BraggNN.to(self.torch_dev)
        self.frames_processed = 0
        self.tq = queue.Queue()
        threading.Thread(target=self.frame_process, daemon=True).start()

    def frame_process(self, ):
        while True:
            pv = self.tq.get()
            frm_id = pv['uniqueId']

            dims = pv['dimension']
            rows = dims[0]['size']
            cols = dims[1]['size']
            frame = pv['value'][0]['ushortValue'].reshape((rows, cols))
            self.frames_processed += 1
            self.tq.task_done()

            # time.sleep(0.1) # mimic processing
            tick = time.time()
            patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=self.psz)
            if patches.shape[0] == 0:
                print("%.3f, %d peaks located in frame %d, %.3fms/frame, %d peaks are too big; %d frames processed so far" % (\
                    time.time(), patches.shape[0], frm_id, elapse, big_peaks, self.frames_processed))
            input_tensor = torch.from_numpy(patches[:, np.newaxis].astype('float32'))
            # todo, infer in a batch fashion in case of out-of-memory
            with torch.no_grad():
                pred = self.BraggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
            peak_locs, big_peaks = pred * self.psz + patch_ori, big_peaks

            elapse = 1000 * (time.time() - tick)
            print("%.3f, %d peaks located in frame %d, %.3fms/frame, %d peaks are too big; %d frames processed so far" % (\
                time.time(), peak_locs.shape[0], frm_id, elapse, big_peaks, self.frames_processed))

    def monitor(self, pv):
        uid = pv['uniqueId']
        self.tq.put(pv.copy())
        print("%.3f received frame %d, processed frames: %d" % (time.time(), uid, self.frames_processed))

def main_monitor():
    c = Channel('pvapy:image')
    c.setMonitorMaxQueueLength(2000)

    client = pvaClient()

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')

    # ToDo check if it is done from server/detector, where streaming gives signal
    time.sleep(1000)
    c.stopMonitor()
    c.unsubscribe('monitor')

# limitation here: the same frame will be get() over and over again till server got the notification
def main_get():
    max_queue_size = -1
    c = Channel('pvapy:image')
    c.setMonitorMaxQueueLength(max_queue_size)

    while True:
        pv = c.get('')
        uid = pv['uniqueId']
        print("received frame %d @ %.3f" % (uid, time.time()))
        dims = pv['dimension']
        rows = dims[0]['size']
        cols = dims[1]['size']
        frame = pv['value'][0]['ushortValue'].reshape((rows, cols))

        print(frame.shape)


if __name__ == '__main__':
    main_monitor()
    # main_get()

