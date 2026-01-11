# LSTMConvSR: Joint Long–Short-Range Modeling via LSTM-First–CNN-Next Architecture for Remote Sensing Image Super-Resolution

<div align="left">

[![paper](https://img.shields.io/badge/Remote%20Sens.%202025,%2017(15),%202745-3A7138)](https://doi.org/10.3390/rs17152745)&nbsp;
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Doswin/LSTMConvSR)

</div>

## Overview

<p align="center">
    <img src="https://github.com/Doswin/LSTMConvSR/blob/main/Overview.svg" width=95%>
<p>

## LSTM-First–CNN-Next

<p align="center">
    <img src="https://github.com/Doswin/LSTMConvSR/blob/main/LSTM-First%E2%80%93CNN-Next.svg" width=95%>
<p>

## Training and Testing

Please use [BasicSR](https://github.com/XPixelGroup/BasicSR), it is an open-source image super-resolution toolbox based on PyTorch.
If you encounter any issues during use, please refer to [MambaIRv2](https://github.com/csguoh/MambaIR) — it is an excellent work and may provide helpful guidance.

## Citation 

If you find this work useful, please consider citing:

```bibtex
@Article{LSTMConvSR,
AUTHOR = {Zhu, Qiwei and Zhang, Guojing and Wang, Xiaoying and Huang, Jianqiang},
TITLE = {LSTMConvSR: Joint Long–Short-Range Modeling via LSTM-First–CNN-Next Architecture for Remote Sensing Image Super-Resolution},
JOURNAL = {Remote Sensing},
VOLUME = {17},
YEAR = {2025},
NUMBER = {15},
ARTICLE-NUMBER = {2745},
URL = {https://www.mdpi.com/2072-4292/17/15/2745},
ISSN = {2072-4292},
ABSTRACT = {The inability of existing super-resolution methods to jointly model short-range and long-range spatial dependencies in remote sensing imagery limits reconstruction efficacy. To address this, we propose LSTMConvSR, a novel framework inspired by top-down neural attention mechanisms. Our approach pioneers an LSTM-first–CNN-next architecture. First, an LSTM-based global modeling stage efficiently captures long-range dependencies via downsampling and spatial attention, achieving 80.3% lower FLOPs and 11× faster speed. Second, a CNN-based local refinement stage, guided by the LSTM’s attention maps, enhances details in critical regions. Third, a top-down fusion stage dynamically integrates global context and local features to generate the output. Extensive experiments on Potsdam, UAVid, and RSSCN7 benchmarks demonstrate state-of-the-art performance, achieving 33.94 dB PSNR on Potsdam with 2.4× faster inference than MambaIRv2.},
DOI = {10.3390/rs17152745}
}
```

## Reference

Some of the codes in this repo are borrowed from:  
- [BasicSR](https://github.com/XPixelGroup/BasicSR)  
- [Vision-LSTM](https://github.com/NX-AI/vision-lstm) 

Thanks to their great work.
