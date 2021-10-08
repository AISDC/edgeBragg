
import time, torch
from pvaccess import Channel
from pvaccess import PvObject
import numpy as np 

from BraggNN import BraggNN
from preprocess import frame_peak_patches_gcenter as frame2patch

class pvaClient:
    def __init__(self, ):
        torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BraggNN  = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2)) # should use the same argu as it in the training.
        mdl_fn = 'models/fc16_8_4_2-sz11.pth'
        self.BraggNN .load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            self.BraggNN = self.BraggNN.to(torch_dev)

    def frame_process(self, frame):
        patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=11)
        input_tensor = torch.from_numpy(patches[:, np.newaxis].astype('float32'))
        with torch.no_grad():
            pred = self.BraggNN.forward(input_tensor).cpu().numpy()
        return pred + patch_ori, big_peaks

    def monitor(self, pv):
        uid = pv['uniqueId']
        print("%.3f received frame %d" % (time.time(), uid, ))
        dims = pv['dimension']
        rows = dims[0]['size']
        cols = dims[1]['size']
        frame = pv['value'][0]['ushortValue'].reshape((rows, cols))

        # time.sleep(1.5) # simulating data processing
        peak_locs, big_peaks = self.frame_process(frame)
        print("%.3f, %d peaks located in frame %d, %d peaks are too big" % (time.time(), peak_locs.shape[0], uid, big_peaks))

def main_monitor():
    max_queue_size = -1
    c = Channel('pvapy:image')
    c.setMonitorMaxQueueLength(max_queue_size)

    client = pvaClient()

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')

    time.sleep(1000)
    c.stopMonitor()

    c.unsubscribe('monitor')

def main_get():
    max_queue_size = 10
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

