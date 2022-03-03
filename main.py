import time, queue, sys, os, multiprocessing
import argparse, logging, yaml
from pvaccess import Channel
import numpy as np 
from multiprocessing import Process, Queue

from inferBraggNN import inferBraggNNtrt, inferBraggNNTorch
from frameProcess import frame_process_worker_func
from asyncWriter import asyncHDFWriter
from pvaClient import pvaClient
from trtUtil import scriptpth2onnx

def main(params):
    logging.info(f"listen on {params['frame']['pvkey']} for frames")
    c = Channel(params['frame']['pvkey'])
    c.setMonitorMaxQueueLength(-1)

    tq_frame = Queue(maxsize=-1) # a task queue for frame processing
    tq_patch = Queue(maxsize=-1) # a task queue for patch processing, i.e., model inference
    rq_peak_write = Queue(maxsize=-1) # results async writter

    # create async peak/result writer
    peak_writer = asyncHDFWriter(params['output']['peaks2file'])
    peak_writer.start()

    # create async frame writer as needed
    if len(params['output']['frame2file']) > 0:
        frame_writer = asyncHDFWriter(params['output']['frame2file'], compression=True)
        frame_writer.start()
    else:
        frame_writer = None

    # initialize pva, it pushes frames into tq_frame
    pva_client = pvaClient(tq_frame=tq_frame)

    # initialize inference engine, which consumes patches from tq_patch
    if params['infer']['tensorrt']:
        onnx_fn = scriptpth2onnx(pth=params['model']['model_fname'], mbsz=params['infer']['mbsz'], psz=params['model']['psz'])
        infer_engine = inferBraggNNtrt(mbsz=params['infer']['mbsz'], onnx_mdl=onnx_fn, tq_patch=tq_patch, peak_writer=peak_writer)
    else:
        infer_engine = inferBraggNNTorch(script_pth=params['model']['model_fname'], tq_patch=tq_patch, peak_writer=peak_writer)
    infer_engine.start()

    # start a pool of processes to digest frame from tq_frame and push patches into tq_patch
    for _ in range(params['frame']['nproc']):
        p = Process(target=frame_process_worker_func, \
                    args=(tq_frame, params['model']['psz'], tq_patch, params['infer']['mbsz'], \
                          params['frame']['offset_recover'], params['frame']['min_intensity'], \
                          params['frame']['max_radius'], frame_writer))
        p.start()

    c.subscribe('monitor', pva_client.monitor)
    c.startMonitor('')
    
    # exit when idle for some seconds or interupted by keyboard
    while True:
        try:
            recv_prog = pva_client.recv_frames
            time.sleep(60)
            if recv_prog == pva_client.recv_frames and tq_frame.qsize()==0 and tq_patch.qsize()==0:
                if args.autoexit != 0:
                    logging.warning("program exits because of silence")
                    for _ in range(params['frame']['nproc']):
                        tq_frame.put((-1, None, None, None, None, None, None))
                    break
                else:
                    logging.warning("Program is alive, no frame came in the past minute.")
        except KeyboardInterrupt:
            for _ in range(params['frame']['nproc']):
                tq_frame.put((-1, None, None, None, None, None, None))
            logging.critical("program exits because KeyboardInterrupt")
            break
        
    time.sleep(1) # give processes seconds to exit
    c.stopMonitor()
    c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edge pipeline for Bragg peak finding')
    parser.add_argument('-gpus',    type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-cfg',     type=str, required=True, help='yaml config file')
    parser.add_argument('-autoexit',type=int, default=0, help='exit when silent for a while')
    parser.add_argument('-verbose', type=int, default=1, help='non-zero to print logs to stdout')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)

    logging.basicConfig(filename='edgeBragg.log', level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)s %(module)s: %(message)s',)
    if args.verbose != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(params)

