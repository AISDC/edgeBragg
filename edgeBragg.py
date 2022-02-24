
import time, queue, multiprocessing, sys, os
import argparse, logging
from pvaccess import Channel
import numpy as np 
from multiprocessing import Process, Queue

from inferBraggNN import inferBraggNNtrt, inferBraggNNTorch
from frameProcess import frame_peak_patches_cv2 as frame2patch
from BraggNN import scriptpth2onnx
from asyncWriter import asyncHDFWriter

class pvaClient:
    def __init__(self, mbsz, psz=15, trt=False, pth='models/feb402.pth'):
        self.psz = 15
        self.patch_tq = Queue(maxsize=-1)
        if trt:
            onnx_fn = scriptpth2onnx(pth, mbsz, psz=psz)
            self.infer_engine = inferBraggNNtrt(mbsz=mbsz, onnx_mdl=onnx_fn, patch_tq=self.patch_tq, \
                                                ofname='inferRes/BraggNN-res.h5')
        else:
            self.infer_engine = inferBraggNNTorch(script_pth=pth, patch_tq=self.patch_tq, \
                                                  ofname='inferRes/BraggNN-res.h5')
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = None
        self.frame_tq = Queue(maxsize=-1)
        self.infer_engine.start()
        
    def monitor(self, pv):
        uid = pv['uniqueId'] # pvaccess.pvaccess.PvObject

        # ignore the 1st empty frame when use sv simulator
        if self.recv_frames is None:
            self.recv_frames = 0
            return 

        if self.base_seq_id is None: self.base_seq_id = uid
        self.recv_frames += 1
        
        # I had problem to pickle PvObject, so just unpack and push to queue
        print(pv['codec'])
        exit()
        frm_id= pv['uniqueId']
        dims  = pv['dimension']
        rows  = dims[0]['size']
        cols  = dims[1]['size']
        frame = pv['value'][0]['ushortValue'].reshape((rows, cols))
        self.frame_tq.put((frm_id, frame))
        logging.info("received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     uid, self.recv_frames, uid - self.base_seq_id + 1, self.frame_tq.qsize()))

def frame_process(frame_tq, psz, patch_tq, mbsz, frame_writer):
    logging.info(f"worker {multiprocessing.current_process().name} starting now")
    patch_list = []
    patch_ori_list = []
    while True:
        try:
            frm_id, frame = frame_tq.get()
        except queue.Empty:
            continue
        except:
            logging.error("Something else of the Queue went wrong")
            continue
        if frm_id < 0:
            break

        tick = time.time()
        patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=psz, angle=frm_id, min_intensity=100)
        patch_list.extend(patches)
        patch_ori_list.extend(patch_ori)

        while len(patch_list) >= mbsz:
            batch_task = (np.array(patch_list[:mbsz])[:,np.newaxis], \
                          np.array(patch_ori_list[:mbsz]).astype(np.float32))
            patch_tq.put(batch_task)
            patch_list = patch_list[mbsz:]
            patch_ori_list = patch_ori_list[mbsz:]
        
        elapse = 1000 * (time.time() - tick)
        logging.info("%d patches cropped from frame %d, %.3fms/frame, %d peaks are too big; "\
                     "%d patches pending infer" % (\
                     len(patches), frm_id, elapse, big_peaks, mbsz*patch_tq.qsize()))
        # back-up raw frames
        frame_writer.append2write({"angle":np.array([frm_id])[None], "frame":frame[None]})

    logging.info(f"worker {multiprocessing.current_process().name} exiting now")

def main_monitor(ch, mbsz, nth):
    c = Channel(ch)
    c.setMonitorMaxQueueLength(-1)

    client = pvaClient(mbsz=mbsz)

    frame_writer = asyncHDFWriter('frames/frames-dump.h5')
    frame_writer.start()
    for _ in range(nth):
        p = Process(target=frame_process, \
                    args=(client.frame_tq, client.psz, client.patch_tq, mbsz, frame_writer))
        p.start()

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')

    # exit when idle for some seconds or interupted by keyboard
    while True:
        try:
            recv_prog = client.recv_frames
            time.sleep(30)
            if recv_prog == client.recv_frames and \
                client.frame_tq.qsize()==0 and \
                client.patch_tq.qsize()==0:
                logging.info("program exits because of silence")
                for _ in range(nth):
                    client.frame_tq.put((-1, None))
                break
        except KeyboardInterrupt:
            for _ in range(nth):
                client.frame_tq.put((-1, None))
            logging.info("program exits because KeyboardInterrupt")
            break

    time.sleep(1) # give processes seconds to exit
    c.stopMonitor()
    c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edge pipeline for Bragg peak finding')
    parser.add_argument('-gpus',    type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-ch',      type=str, default='1id-ADSim2:Pva1:Image', help='pva channel name')
    parser.add_argument('-nth',     type=int, default=1, help='number of threads for frame processes')
    parser.add_argument('-mbsz',    type=int, default=1024, help='inference batch size')
    parser.add_argument('-verbose', type=int, default=1, help='non-zero to print logs to stdout')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.basicConfig(filename='edgeBragg.log', level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)-8s %(message)s',)
    if args.verbose != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main_monitor(ch=args.ch, nth=args.nth, mbsz=args.mbsz)

