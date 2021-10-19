# edgeHEDM
HEDM on the edge near data source

This repo hosts code to localize (with sub-pixel accuracy) Bragg peaks from X-ray diffraction frames streamed from EPICS-enabled area detector.
[BraggNN](https://arxiv.org/abs/2008.08198) trained using code in this [repo](https://github.com/lzhengchun/BraggNN) or remote data center AI-system using this distributed [workflow](https://arxiv.org/abs/2105.13967), is used to localize Bragg peaks faster than conventional psuedo-Voigt.

For debug and evaluation purpose, one can also use `daq-simu-pva.py` to simulate data (of given) streamed from the area detector.

**Note**: based on our testing, the streaming framework, i.e., EPICS+pavpy, starts losing data randomly and silently when throughput is too high. For example, we start experiencing frame loss when FPS is greater than 10 for frames with 2048x2048 @16 bits. But it's easy to handle 500 FPS for 256x256 @ 16bit frames. These numbers also depend on the headware spec of the system running client side.
We are currently investigating the root cause of frame loss and tring to fix it. One can use this framework for low throughput streaming process.
