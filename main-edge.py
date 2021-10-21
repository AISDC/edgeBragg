
import time, queue, threading, sys, os
import torch, argparse, logging
from pvaccess import Channel
from pvaccess import PvObject
import numpy as np 

from BraggNN import BraggNN
from preprocess import frame_peak_patches_gcenter as frame2patch

class pvaClient:
    def __init__(self, nth=1):
        self.psz = 15
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BraggNN  = BraggNN(imgsz=self.psz, fcsz=(16, 8, 4, 2)) # should use the same argu as it in the training.
        mdl_fn = 'models/fc16_8_4_2-sz15.pth'
        self.BraggNN .load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            self.BraggNN = self.BraggNN.to(self.torch_dev)
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = 0
        self.tq = queue.Queue(maxsize=-1)
        self.thr_exit = 0

        for _ in range(nth):
            threading.Thread(target=self.frame_process, daemon=True).start()

    def frame_process(self, ):
        while self.thr_exit == 0:
            try:
                pv = self.tq.get(block=True, timeout=1)
            except queue.Empty:
                continue
            except:
                logging.error("Something else of the Queue went wrong")
                continue

            frm_id= pv['uniqueId']
            dims  = pv['dimension']
            rows  = dims[0]['size']
            cols  = dims[1]['size']
            frame = pv['value'][0]['ushortValue'].reshape((rows, cols))
            self.tq.task_done()

            tick = time.time()
            patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=self.psz)
            if patches.shape[0] > 0:
                input_tensor = torch.from_numpy(patches[:, np.newaxis].astype('float32'))
                # todo, infer in a batch fashion in case of out-of-memory
                with torch.no_grad():
                    pred = self.BraggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
                peak_locs, big_peaks = pred * self.psz + patch_ori, big_peaks

            self.frames_processed += 1 # has race condition
            elapse = 1000 * (time.time() - tick)
            logging.info("[%.3f] %d peaks located in frame %d, %.3fms/frame, %d peaks are too big; %d out of %d frames processed so far" % (\
                        time.time(), patches.shape[0], frm_id, elapse, big_peaks, self.frames_processed, self.recv_frames))

        logging.info(f"worker thread {threading.get_ident() } exiting now")

    def monitor(self, pv):
        uid = pv['uniqueId']
        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1
        self.tq.put(pv.copy())
        logging.info("[%.3f] received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     time.time(), uid, self.recv_frames, uid - self.base_seq_id + 1, self.tq.qsize()))

def main_monitor(ch, nth):
    c = Channel(ch)
    c.setMonitorMaxQueueLength(-1)

    client = pvaClient(nth)

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')

    # ToDo check if it is done from server/detector, where streaming gives signal
    time.sleep(2)

    # exit when idle for 10 seconds
    while True:
        recv_prog = client.recv_frames
        proc_prog = client.frames_processed
        time.sleep(10)
        if recv_prog == client.recv_frames and proc_prog == client.frames_processed:
            logging.info("program exits because of silence")
            break
    client.tq.join()
    client.thr_exit = 1
    time.sleep(2) # give threads seconds to exit

    c.stopMonitor()
    c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edge pipeline for Bragg peak finding')
    parser.add_argument('-gpus', type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-ch',   type=str, default='pvapy:image', help='pva channel name')
    parser.add_argument('-nth',  type=int, default=2, help='number of threads for frame processes')
    parser.add_argument('-terminal',  type=int, default=0, help='non-zero to print logs to stdout')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.basicConfig(filename='edgeBragg.log', level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)-8s %(message)s',)
    if args.terminal != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main_monitor(ch=args.ch, nth=args.nth)

