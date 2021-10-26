#! /bin/bash
PY=/usr/bin/python3
PY=/homes/zhengchun.liu/usr/miniconda3/envs/trt/bin/python

$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=8  -samples=2560
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=16 -samples=5120
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=32
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=64
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=128
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=256
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz11.pth -psz=11 -mbsz=512

# echo " "
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=8  -samples=2560
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=16 -samples=5120
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=32
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=64
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=128
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=256
$PY ./bm_tensorRT_infer.py -psz=11 -mdl=models/fc16_8_4_2-sz11.pth -mbsz=512

echo " "
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=8  -samples=2560
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=16 -samples=5120
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=32
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=64
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=128
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=256
$PY ./bm_BraggNN.py -mdl=models/fc16_8_4_2-sz15.pth -psz=15 -mbsz=512

echo " "

$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=8  -samples=2560
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=16 -samples=5120
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=32
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=64
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=128
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=256
$PY ./bm_tensorRT_infer.py -psz=15 -mdl=models/fc16_8_4_2-sz15.pth -mbsz=512
