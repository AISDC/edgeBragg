
import time, queue, sys, os, multiprocessing
import argparse, logging
from pvaccess import Channel
# from pvaccess import PvObject
import numpy as np 
from multiprocessing import Process, Queue

from inferBraggNN import inferBraggNNtrt, inferBraggNNTorch
from frameProcess import frame_peak_patches_cv2 as frame2patch
from codecAD import CodecAD
from BraggNN import pth2onnx

class pvaClient:
    def __init__(self, mbsz, psz=15, trt=False, pth='models/fc16_8_4_2-sz15.pth'):
        self.psz = 15
        self.patch_tq = Queue(maxsize=-1)
        if trt:
            mdl_fn = pth2onnx(pth, mbsz, psz=psz)
            self.infer_engine = inferBraggNNtrt(mbsz=mbsz, onnx_mdl=mdl_fn, patch_tq=self.patch_tq)
        else:
            self.infer_engine = inferBraggNNTorch(pth_mdl=pth, psz=psz,  patch_tq=self.patch_tq, fcsz=(16, 8, 4, 2))
        self.frames_processed = 0
        self.base_seq_id = None
        self.recv_frames = None
        self.frame_tq = Queue(maxsize=-1)
        self.codecAD = CodecAD()
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
        frm_id= pv['uniqueId']
        data_codec = pv['value'][0]['ubyteValue'] # will broken for uncoded, non-ubyte data
        compressed = pv["compressedSize"]
        uncompressed = pv["uncompressedSize"]
        codec = pv["codec"]
        dims  = pv['dimension']
        rows  = dims[0]['size']
        cols  = dims[1]['size']
        self.frame_tq.put((frm_id, data_codec, compressed, uncompressed, codec, rows, cols))
        logging.info("[%.3f] received frame %d, total frame received: %d, should have received: %d; %d frames pending process" % (\
                     time.time(), uid, self.recv_frames, uid - self.base_seq_id + 1, self.frame_tq.qsize()))

def frame_process(frame_tq, codecAD, psz, patch_tq, mbsz):
    patch_list = []
    while True:
        try:
            frm_id, data_codec, compressed, uncompressed, codec, rows, cols = frame_tq.get()
        except queue.Empty:
            continue
        except:
            logging.error("Something else of the Queue went wrong")
            continue

        if frm_id < 0:
            break

        dec_tick = time.time()
        if codecAD.decompress(data_codec, codec, compressed, uncompressed):
            data = codecAD.getData()
            dec_time = 1000 * (time.time() - dec_tick)
            logging.info("[%.3f] frame %d has been decoded in %.2f ms, compress ratio is %.1f" % (\
                            time.time(), frm_id, dec_time, codecAD.getCompressRatio()))
        else:
            logging.error("data is not compressed!")
            data = data_codec

        frame = data.reshape((rows, cols))

        tick = time.time()
        patches, patch_ori, big_peaks = frame2patch(frame=frame, psz=psz)
        patch_list += patches

        while len(patch_list) >= mbsz:
            patch_tq.put(np.array(patch_list[:mbsz])[:,np.newaxis])
            patch_list = patch_list[mbsz:]
        
        elapse = 1000 * (time.time() - tick)
        logging.info("[%.3f] %d patches cropped from frame %d, %.3fms/frame, %d peaks are too big; "\
                     "%d patches pending infer" % (\
                     time.time(), patch_ori.shape[0], frm_id, elapse, big_peaks, mbsz*patch_tq.qsize()))
    logging.info(f"worker {multiprocessing.current_process().name} exiting now")

def main_monitor(ch, mbsz, nth):
    c = Channel(ch)
    c.setMonitorMaxQueueLength(-1)

    client = pvaClient(mbsz=mbsz, trt=True)

    for _ in range(nth):
        p = Process(target=frame_process, \
                    args=(client.frame_tq, client.codecAD, client.psz, client.patch_tq, mbsz))
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
                    client.frame_tq.put((-1, None, None, None, None, None, None))
                break
        except KeyboardInterrupt:
            for _ in range(nth):
                client.frame_tq.put((-1, None, None, None, None, None, None))
            logging.info("program exits because KeyboardInterrupt")
            break
        
    time.sleep(1) # give processes seconds to exit
    c.stopMonitor()
    c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edge pipeline for Bragg peak finding')
    parser.add_argument('-gpus',    type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-ch',      type=str, default='13SIM1:Pva1:Image', help='pva channel name')
    parser.add_argument('-nth',     type=int, default=6, help='number of threads for frame processes')
    parser.add_argument('-mbsz',    type=int, default=1024, help='inference batch size')
    parser.add_argument('-verbose', type=int, default=0, help='non-zero to print logs to stdout')

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

