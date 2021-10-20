
# import time, torch
import time
import argparse
from pvaccess import Channel
import numpy as np 

from BraggNN import BraggNN
from preprocess import frame_peak_patches_gcenter as frame2patch

class pvaClient:
    def __init__(self, sim_processing_time=0, n_skip_frames=0):
        self.last_uid = None
        self.n_missed = 0
        self.n_received = None
        self.sim_processing_time = sim_processing_time
        self.n_skip_frames = n_skip_frames
        #self.process_setup()

    def process_setup(self):
        self.psz = 15
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BraggNN  = BraggNN(imgsz=self.psz, fcsz=(16, 8, 4, 2)) # should use the same argu as it in the training.
        mdl_fn = 'models/fc16_8_4_2-sz15.pth'
        self.BraggNN .load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
             self.BraggNN = self.BraggNN.to(self.torch_dev)

    def frame_process(self, frame):
        now = time.time()
        uid = frame['uniqueId']
        time.sleep(self.sim_processing_time)
        print('processed frame %d @ %.3f' % (uid, now))

        # patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=self.psz)
        # input_tensor = torch.from_numpy(patches[:, np.newaxis].astype('float32'))
        # # todo, infer in a batch fashion in case of out-of-memory
        # with torch.no_grad():
        #     pred = self.BraggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
        # return pred * self.psz + patch_ori, big_peaks

    def monitor(self, pv):
        now = time.time()
        # Ignore first frame, will be empty
        if self.n_received is None:
            self.n_received = 0
            return

        uid = pv['uniqueId']
        if uid == self.last_uid:
            # Bug in epics libraries (deadband plugin)
            #print("Received frame %d more than once" % uid)
            return
        self.n_received += 1
        if self.last_uid is not None:
            n_missed = int((uid - self.last_uid - 1 - self.n_skip_frames) / (self.n_skip_frames+1))
            if n_missed:    
                self.n_missed += n_missed
                print("Lost %s frames @ uid %s (total missed: %s)" % (n_missed,uid,self.n_missed))
        if self.n_received and self.n_received % 100 == 0:
            print("%.3f received frame %d (total received: %s, total missed: %s)" % (time.time(), uid, self.n_received, self.n_missed))
        self.last_uid = uid
        self.frame_process(pv)
        return

        dims = pv['dimension']
        rows = dims[0]['size']
        cols = dims[1]['size']
        frame = pv['value'][0]['ubyteValue'].reshape((rows, cols))

        # further optimization: 
        # (1) overlap preporcess with BraggNN inference (on GPU)
        # (2) more consumers to frames
        peak_locs, big_peaks = self.frame_process(frame)

        elapse = 1000 * (time.time() - tick)
        print("%.3f, %d peaks located in frame %d, %.3fms/frame, %d peaks are too big" % (\
              time.time(), peak_locs.shape[0], uid, elapse, big_peaks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='client simulation for data streaming from detector using EPICS')
    parser.add_argument('-cn', type=str, default='pvapy:image', help='server channel name')
    parser.add_argument('-qs', type=int, default=10000, help='queue size')
    parser.add_argument('-spt', type=float, default=0.1, help='simulated processing time in seconds')
    parser.add_argument('-sf', type=int, default=0, help='specifies how many frames to skip')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    pv_request = ''
    if args.sf > 0:
        pv_request = 'uniqueId[deadband=abs:%s],dimension,value' % (args.sf+1)
    c = Channel(args.cn)
    client = pvaClient(args.spt, args.sf)
    c.setMonitorMaxQueueLength(args.qs)
    time.sleep(1)
    c.monitor(client.monitor, pv_request)
    time.sleep(1000)
    c.stopMonitor()



