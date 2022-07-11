#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs/configs_resnest/resnest_super_cwd.py 2 --work-dir nfs/mobileAI2022/distill_final_grad_poly_resnest_super/cwd_3e-1_bilinear

bash ./tools/dist_train.sh projects/configs/configs_resnest/resnest_super_cwd.py 2 --work-dir nfs/mobileAI2022/distill_final_grad_poly_resnest_super/cwd_1e-1_bilinear --options 'model.distill_loss.loss_weight=0.1'

