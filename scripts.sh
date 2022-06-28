#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs/configs_distill/dynamic_crop_cwd_swinl_align.py 2 --work-dir nfs/mobileAI2022/distill/dynamic_crop_cwd_4_swinl_align --options 'model.distill_loss.loss_weight=4'

bash ./tools/dist_train.sh projects/configs/configs_distill/dynamic_crop_cwd_swinl_align.py 2 --work-dir nfs/mobileAI2022/distill/dynamic_crop_cwd_5_swinl_align --options 'model.distill_loss.loss_weight=5'

bash ./tools/dist_train.sh projects/configs/configs_distill/dynamic_crop_cwd_swinl_align.py 2 --work-dir nfs/mobileAI2022/distill/dynamic_crop_cwd_6_swinl_align --options 'model.distill_loss.loss_weight=6'


