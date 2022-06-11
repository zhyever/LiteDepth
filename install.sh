#!/usr/bin/env bash

conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
cd Monocular-Depth-Estimation-Toolbox && pip install -e . && cd ..
cd mmclassification && pip install -e . && cd ..
pip install timm