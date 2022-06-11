import mmcv
import os
import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='depth test (and eval) a model')
    parser.add_argument('input_path', help='test config file path')
    parser.add_argument('output_path', help='checkpoint file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    file_names = os.listdir(args.input_path)

    mmcv.mkdir_or_exist(args.output_path)
    
    for name in file_names:

        file_path = os.path.join(args.input_path, name)
        npy_file = np.load(file_path).astype(np.uint16)[0,:,:]

        filename = name[:-4]
        filename = filename + '.png'

        # mmcv.imwrite(npy_file, os.path.join(args.output_path, filename))
        cv2.imwrite(os.path.join(args.output_path, filename), npy_file, [cv2.IMWRITE_PNG_COMPRESSION, 0])
