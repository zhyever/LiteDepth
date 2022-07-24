# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np
import torch

@LOSSES.register_module()
class PairMSELoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0, sample_ratio=0.15):
        super().__init__()
        self.loss_weight = loss_weight
        self.sample_ratio = sample_ratio

    def select_index(self, H, W):
        valid_width = W
        valid_height = H
        num = valid_width * valid_height

        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)

        p1_x = p1 % W
        p1_y = (p1 / W).astype(np.int)

        p2_x = p2 % W
        p2_y = (p2 / W).astype(np.int)

        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y}

        return p123

    def forward(self, gt_depth, pred_depth, select=True):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        p12 = self.select_index(gt_depth.shape[-2], gt_depth.shape[-1])

        p1_x = p12['p1_x']
        p1_y = p12['p1_y']
        p2_x = p12['p2_x']
        p2_y = p12['p2_y']

        gt_source = gt_depth[:, :, p1_y, p1_x]
        gt_target = gt_depth[:, :, p2_y, p2_x]

        gt_source_masked = gt_source
        gt_target_masked = gt_target

        pred_source = pred_depth[:, :, p1_y, p1_x]
        pred_target = pred_depth[:, :, p2_y, p2_x]

        gt_diff = gt_source_masked - gt_target_masked
        pred_diff = pred_source - pred_target

        gt_diff = gt_diff.clone()
        pred_diff = pred_diff.clone()
        gt_diff[torch.isnan(gt_diff)] = 0.
        pred_diff[torch.isnan(gt_diff)] = 0.
        gt_diff = gt_diff.clone()
        pred_diff = pred_diff.clone()
        gt_diff[torch.isinf(gt_diff)] = 0.
        pred_diff[torch.isinf(gt_diff)] = 0.

        loss = torch.abs(gt_diff - pred_diff)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.shape[0] * 0.25):]

        loss = torch.mean(loss) if loss.shape[0] > 0 else gt_diff.new_tensor(0.0)

        loss =  self.loss_weight * loss
        return loss