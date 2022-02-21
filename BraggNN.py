import logging, torch

def scriptpth2onnx(pth, mbsz, psz):
    model = torch.jit.load(pth, map_location='cpu')
    if psz != model.input_psz.item():
        logging.error(f"The provided torchScript model is trained for patch size of {model.input_psz.item()}!")

    dummy_input = torch.randn(mbsz, 1, psz, psz, dtype=torch.float32, device='cpu')

    input_names  = ('patch', )
    output_names = ('ploc',  )

    onnx_fn = pth.replace(".pth", ".onnx")
    torch.onnx.export(model, dummy_input, onnx_fn, verbose=False, \
                      input_names=input_names, output_names=output_names)
    return onnx_fn