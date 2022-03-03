# edgeHEDM
Realtime Bragg peak analysis on the edge near data source

This repo hosts code to localize (with sub-pixel accuracy) Bragg peaks from X-ray diffraction frames streamed from EPICS-enabled area detector.
[BraggNN](https://arxiv.org/abs/2008.08198) trained using code in this [repo](https://github.com/lzhengchun/BraggNN) or remote data center AI-system using this distributed [workflow](https://arxiv.org/abs/2105.13967), is used to localize Bragg peaks faster than conventional psuedo-Voigt.

For debug and evaluation purpose, one can also use `daq-simu-pva.py` to simulate data (of given) streamed from the area detector.

