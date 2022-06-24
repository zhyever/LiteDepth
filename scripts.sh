#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs_aug/dynamic_crop_200e.py 2 --work-dir nfs/mobileAI2022/crop_strategy/dynamic_crop_200e
bash ./tools/dist_train.sh projects/configs_aug/crop_384x512_200e.py 2 --work-dir nfs/mobileAI2022/crop_strategy/crop_384x512_200e
bash ./tools/dist_train.sh projects/configs_aug/wocrop_200e.py 2 --work-dir nfs/mobileAI2022/crop_strategy/crop_384x512_200e


