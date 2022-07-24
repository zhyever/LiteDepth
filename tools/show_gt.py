
import os
from PIL import Image
import numpy as np
from depth.utils import colorize
import mmcv

# path = '/home/zhyever/zhenyuli/code/MobileAI2022_model/online_results'
path = 'data/local_val/depth'

output_path = 'nfs/mobileAI2022/val_gt_show'
mmcv.mkdir_or_exist(output_path)

file_name = os.listdir(path)

for item in file_name:
    depth_map_path = os.path.join(path, item)
    depth = np.asarray(Image.open(depth_map_path), dtype=np.float32)
    # depth[depth==0] = 40000
    # depth_colored = colorize(depth[np.newaxis, :], vmin=0, vmax=40000)
    depth_colored = colorize(depth[np.newaxis, :], vmin=None, vmax=None)
    mmcv.imwrite(depth_colored.squeeze(), os.path.join(output_path, item))