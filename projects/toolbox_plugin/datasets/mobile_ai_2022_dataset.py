import os
import mmcv

import numpy as np
from PIL import Image
from mmcv.utils import print_log
from prettytable import PrettyTable
from collections import OrderedDict

from depth.datasets import DATASETS, CustomDepthDataset

@DATASETS.register_module()
class MobileAI2022Dataset(CustomDepthDataset):
    r"""MobileAI2022Dataset Dataset.

    """

    # waiting to be done
    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.depth_scale).astype(np.uint16)
        return results


    def pre_eval(self, preds, indices):
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map = os.path.join(self.depth_path, self.img_infos[index]['ann']['depth_map'])

            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32)
            depth_map_gt = np.expand_dims(depth_map_gt, axis=0) # raw depth gt

            # scale pred meters back to millimeters
            eval = metrics(pred * self.depth_scale, depth_map_gt)
            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds


    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            raise NotImplementedError
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)

        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 2
        for i in range(num_table):
            names = ret_metric_names[i*2: i*2 + 2]
            values = ret_metric_values[i*2: i*2 + 2]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results


def pre_eval_to_metrics(pre_eval_results):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics['rmse'] = np.nanmean(pre_eval_results[0])
    ret_metrics['sirmse'] = np.nanmean(pre_eval_results[1])

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }

    return ret_metrics

def metrics(img, target):
    r"""
    From https://github.com/aiff22/MAI-2021-Workshop/blob/main/depth_estimation/losses_python.py
    """
    # If the input tensors are not floating-point ones, first convert them to the corresponding format:
    # img = np.asarray(img, dtype=np.float)
    # target = np.asarray(target, dtype=np.float)

    # NOTE: rmse
    img[img < 1] = 1
    target[target < 1] = 1 # invalid depth value is 0
    mask = np.asarray(target > 1, dtype=int)

    diff = (target - img) * mask / 1000.0   # mapping the distance from millimeters to meters
    num_pixels = float(np.sum(mask > 0))

    rmse = np.sqrt(np.sum(np.square(diff)) / num_pixels)

    # NOTE: sirmse
    log_diff = (np.log(img) - np.log(target)) * mask
    num_pixels = float(np.sum(mask > 0))

    sirmse = np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

    return rmse, sirmse

