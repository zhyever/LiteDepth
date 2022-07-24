#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs/configs_baseline/basemodel_crop_gradloss_vnl_robust.py  2 --work-dir nfs/mobileAI2022_final/robust_weight/weight_5e-1_wolog --options 'model.decode_head.loss_robust.log=False' 'model.decode_head.loss_robust.loss_weight=0.5'

bash ./tools/dist_train.sh projects/configs/configs_baseline/basemodel_crop_gradloss_vnl_robust.py  2 --work-dir nfs/mobileAI2022_final/robust_weight/weight_6e-1_wolog --options 'model.decode_head.loss_robust.log=False' 'model.decode_head.loss_robust.loss_weight=0.6'
