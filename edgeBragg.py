import time, queue, sys, os, multiprocessing
import argparse, logging, yaml
from pvaccess import Channel
import numpy as np 
from multiprocessing import Process, Queue

from frameProcess import frame_process_worker_func
from asyncWriter import asyncHDFWriter
from pvaClient import pvaClient

def main_monitor(params):
    logging.info(f"listen on {params['frame']['pvkey']} for frames")
    c = Channel(params['frame']['pvkey'])
    c.setMonitorMaxQueueLength(-1)

    client = pvaClient(mbsz=params['infer']['mbsz'], psz=params['model']['psz'],\
                       trt=params['infer']['tensorrt'], pth=params['model']['model_fname'],\
                       ofname=params['output']['peaks2file'])
    if len(params['output']['frame2file']) > 0:
        frame_writer = asyncHDFWriter(params['output']['frame2file'])
        frame_writer.start()
    else:
        frame_writer = None
    for _ in range(params['frame']['nproc']):
        p = Process(target=frame_process_worker_func, \
                    args=(client.frame_tq, params['model']['psz'],\
                          client.patch_tq, params['infer']['mbsz'], \
                          params['frame']['offset_recover'], \
                          params['frame']['min_intensity'], frame_writer))
        p.start()

    c.subscribe('monitor', client.monitor)
    c.startMonitor('')
    # exit when idle for some seconds or interupted by keyboard
    while True:
        try:
            recv_prog = client.recv_frames
            time.sleep(60)
            if recv_prog == client.recv_frames and \
                client.frame_tq.qsize()==0 and \
                client.patch_tq.qsize()==0:
                logging.info("program exits because of silence")
                for _ in range(params['frame']['nproc']):
                    client.frame_tq.put((-1, None, None, None, None, None, None))
                break
        except KeyboardInterrupt:
            for _ in range(params['frame']['nproc']):
                client.frame_tq.put((-1, None, None, None, None, None, None))
            logging.info("program exits because KeyboardInterrupt")
            break
        
    time.sleep(1) # give processes seconds to exit
    c.stopMonitor()
    c.unsubscribe('monitor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edge pipeline for Bragg peak finding')
    parser.add_argument('-gpus',    type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-cfg',     type=str, default=None, help='yaml config file')
    parser.add_argument('-verbose', type=int, default=1, help='non-zero to print logs to stdout')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)

    logging.basicConfig(filename='edgeBragg.log', level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)-8s %(message)s',)
    if args.verbose != 0:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main_monitor(params)

