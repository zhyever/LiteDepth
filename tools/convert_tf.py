# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from depth import __version__
from depth.apis import set_random_seed, train_depther
from depth.datasets import build_dataset
from depth.models import build_depther, build_depther
from depth.utils import collect_env, get_root_logger

from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from depth.datasets import build_dataloader, build_dataset

import numpy as np

# DO NOT COMMENT THIS LINE (IT IS DISABLING GPU)!
# WHEN COMMENTED, THE RESULTING TF MODEL WILL HAVE INCORRECT LAYER FORMAT
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import onnx
from onnx_tf.backend import prepare

import tensorflow as tf

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train a depthor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--output-path', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--convert', action='store_true')
    parser.add_argument(
        '--test-model', action='store_true')
    parser.add_argument(
        '--test-output-path', help='the checkpoint file to load weights from')
    args = parser.parse_args()
    return args

def convert_model(model, args):
    # Converting model to ONNX
    for _ in model.modules():
        _.training = False

    sample_input = torch.randn(1, 3, 480, 640)
    input_nodes = ['input']
    output_nodes = ['output']

    # torch.onnx.export(model, sample_input, "model.onnx", export_params=True, input_names=input_nodes, output_names=output_nodes)
    torch.onnx.export(model, sample_input, "nfs/model.onnx", export_params=True, input_names=input_nodes, output_names=output_nodes, opset_version=11)

    # Converting model to Tensorflow

    onnx_model = onnx.load("nfs/model.onnx")
    output = prepare(onnx_model)
    output.export_graph("nfs/tf_model/")

    # Exporting the resulting model to TFLite

    converter = tf.lite.TFLiteConverter.from_saved_model("nfs/tf_model")
    tflite_model = converter.convert()

    # model.tflite
    open(args.output_path, "wb").write(tflite_model)


def tf_test_model(image, args, model, filename):
    # conduct test on this tflite model
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.output_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # have a test
    input_data = np.array(image.numpy(), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tf_result = interpreter.get_tensor(output_details[0]['index'])

    torch_result = model(image)
    torch_result = torch_result.detach().cpu().numpy()

    print("Error of converted model: {}".format(np.mean(torch_result - tf_result)))

    plt.subplot(1, 3, 1)
    img = plt.imread(filename)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(torch_result[0, 0, :, :])
    plt.subplot(1, 3, 3)
    plt.imshow(tf_result[0, 0, :, :])
    plt.savefig(args.test_output_path)
    

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)


    if args.load_from is not None:
        cfg.load_from = args.load_from

    dataset = build_dataset(cfg.data.test)

    model = build_depther(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.eval()

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        image = data['img'][0]
        filename = data['img_metas'][0]._data[0][0]['filename']
        break

    checkpoint = load_checkpoint(model, args.load_from, map_location='cpu')

    if args.convert:
        convert_model(model, args)
    if args.test_model:
        tf_test_model(image, args, model, filename)
    
if __name__ == '__main__':
    main()
