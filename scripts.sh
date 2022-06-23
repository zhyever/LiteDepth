#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs_select_strategy/cwd_swinl.py 2 --work-dir nfs/mobileAI2022/distill_strategy/cwd_swinl_2 --options 'model.distill_loss.loss_weight=2'

bash ./tools/dist_train.sh projects/configs_select_strategy/cwd_swinl.py 2 --work-dir nfs/mobileAI2022/distill_strategy/cwd_swinl_5e-1 --options 'model.distill_loss.loss_weight=0.5'


