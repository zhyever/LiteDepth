#!/usr/bin/env bash

bash ./tools/dist_train.sh projects/configs/configs_arch/tf_mobile_small_mini_decoder_light_poly_gradloss_ssimloss.py 2 --work-dir nfs/mobileAI2022/arch/tf_mobile_small_mini_decoder_light_merge_poly_gradloss5e-1_ssim1 --options 'model.decode_head.loss_ssim.loss_weight=1'

bash ./tools/dist_train.sh projects/configs/configs_arch/tf_mobile_small_mini_decoder_light_poly_gradloss_ssimloss.py 2 --work-dir nfs/mobileAI2022/arch/tf_mobile_small_mini_decoder_light_merge_poly_gradloss5e-1_ssim8e-1 --options 'model.decode_head.loss_ssim.loss_weight=0.8'

bash ./tools/dist_train.sh projects/configs/configs_arch/tf_mobile_small_mini_decoder_light_poly_gradloss_ssimloss.py 2 --work-dir nfs/mobileAI2022/arch/tf_mobile_small_mini_decoder_light_merge_poly_gradloss5e-1_ssim2 --options 'model.decode_head.loss_ssim.loss_weight=2'
