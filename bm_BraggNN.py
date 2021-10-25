
from BraggNN import BraggNN
import torch, argparse, os, time, sys, logging
from torch.utils.data import DataLoader, Dataset
import numpy as np
import onnxruntime as rt

class BraggNNDataset(Dataset):
    def __init__(self, samples=10240, psz=11):
        self.patches = torch.rand(samples, 1, psz, psz)
        self.peakLoc = torch.rand(samples, 2)
        self.psz = psz

    def __getitem__(self, idx):
        return self.patches[idx], self.peakLoc[idx]

    def __len__(self):
        return self.patches.shape[0]

def main(args):
    ds = BraggNNDataset(args.samples, args.psz)
    mb_data_iter = DataLoader(dataset=ds, batch_size=args.mbsz, shuffle=True,\
                              num_workers=2, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)

    model = BraggNN(imgsz=ds.psz, fcsz=(16, 8, 4, 2))
    model.load_state_dict(torch.load(args.mdl, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(torch_devs)
        logging.info("%d GPUs detected, one will be used for the DNN" % gpus)

    pred, gt = [], []
    inference_tick = time.time()
    time_comp = 0
    for X_mb, y_mb in mb_data_iter:
        X_mb_dev = X_mb.to(torch_devs)
        it_comp_tick = time.time()
        with torch.no_grad():
                pred_val = model.forward(X_mb_dev).cpu().numpy()
        time_comp += 1000 * (time.time() - it_comp_tick)

        pred.append(pred_val)
        gt.append(y_mb.numpy())
    time_on_inference = 1000 * (time.time() - inference_tick)

    pred = np.concatenate(pred, axis=0)
    gt   = np.concatenate(gt,   axis=0)
    rate = time_on_inference / gt.shape[0]
    logging.info("Inference for %d samples using mbsz of %d, took %.3f seconds (comp=%.3fs, %.3f ms/batch %.3f ms/sample)" % (\
                 gt.shape[0], args.mbsz, time_on_inference*1e-3, time_comp*1e-3, time_on_inference/(len(mb_data_iter)), rate))

def main_jit(args):
    ds = BraggNNDataset(args.samples, args.psz)
    mb_data_iter = DataLoader(dataset=ds, batch_size=args.mbsz, shuffle=True,\
                              num_workers=2, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)

    model = BraggNN(imgsz=ds.psz, fcsz=(16, 8, 4, 2))
    model.load_state_dict(torch.load(args.mdl, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(torch_devs)
        logging.info("%d GPUs detected, one will be used for the DNN" % gpus)

    model.eval()
    # dummy_input = torch.rand(args.mbsz, 1, args.psz, args.psz).cuda()
    script_cell_gpu = torch.jit.script(model)

    pred, gt = [], []
    inference_tick = time.time()
    time_comp = 0
    for X_mb, y_mb in mb_data_iter:
        X_mb_dev = X_mb.to(torch_devs)
        it_comp_tick = time.time()
        pred_val = script_cell_gpu(X_mb_dev).cpu().numpy()
        time_comp += 1000 * (time.time() - it_comp_tick)

        pred.append(pred_val)
        gt.append(y_mb.numpy())
    time_on_inference = 1000 * (time.time() - inference_tick)

    pred = np.concatenate(pred, axis=0)
    gt   = np.concatenate(gt,   axis=0)
    rate = time_on_inference / gt.shape[0]
    logging.info("Inference for %d samples using mbsz of %d, took %.3f seconds (comp=%.3fs, %.3f ms/batch %.3f ms/sample)" % (\
                 gt.shape[0], args.mbsz, time_on_inference*1e-3, time_comp*1e-3, time_on_inference/(len(mb_data_iter)), rate))

def pth2onnx(args):
    model = BraggNN(imgsz=args.psz, fcsz=(16, 8, 4, 2))

    model.load_state_dict(torch.load(args.mdl, map_location=torch.device('cpu')))
    # model = model.cuda()
    dummy_input = torch.randn(args.mbsz, 1, args.psz, args.psz, dtype=torch.float32, device='cpu')

    input_names  = ('patch', )
    output_names = ('ploc',  )

    onnx_fn = args.mdl.replace(".pth", ".onnx")
    torch.onnx.export(model, dummy_input, onnx_fn, verbose=True, \
                      input_names=input_names, output_names=output_names)
    return onnx_fn

def main_onnx(args):
    if os.path.exists(args.mdl.replace(".pth", ".onnx")):
        onnx_fn = args.mdl.replace(".pth", ".onnx")
    else:
        logging.info("convert %s to onnx ..." % (args.mdl))
        onnx_fn = pth2onnx(args)

    sess = rt.InferenceSession(onnx_fn)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    ds = BraggNNDataset(args.samples, args.psz)
    mb_data_iter = DataLoader(dataset=ds, batch_size=args.mbsz, shuffle=True,\
                              num_workers=2, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)
    pred, gt = [], []
    inference_tick = time.time()
    time_comp = 0
    for X_mb, y_mb in mb_data_iter:
        #X_mb_dev = X_mb.cuda()
        it_comp_tick = time.time()
        pred_val = sess.run([label_name], {input_name: X_mb.numpy()})[0]
        time_comp += 1000 * (time.time() - it_comp_tick)

        pred.append(pred_val)
        gt.append(y_mb.numpy())
    time_on_inference = 1000 * (time.time() - inference_tick)

    pred = np.concatenate(pred, axis=0)
    gt   = np.concatenate(gt,   axis=0)
    rate = time_on_inference /  gt.shape[0]
    logging.info("Inference for %d samples using mbsz of %d, took %.3f seconds (comp=%.3fs, %.3f ms/batch %.3f ms/sample)" % (\
                 gt.shape[0], args.mbsz, time_on_inference*1e-3, time_comp*1e-3, time_on_inference/(len(mb_data_iter)), rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
    parser.add_argument('-gpus',   type=str, default="0", help='the GPU to use')
    parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
    parser.add_argument('-psz',    type=int, default=15, help='input size')
    parser.add_argument('-samples',type=int, default=10240, help='sample size')
    parser.add_argument('-ifn',    type=str, default=None, help='input h5 file')
    parser.add_argument('-mdl',    type=str, default='models/fc16_8_4_2-sz15.pth', help='model weights')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(filename='inference.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)
    # main_jit(args)
    # main_onnx(args)
    # pth2onnx(args)