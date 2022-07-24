# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np
import torch

@LOSSES.register_module()
class SiRMSELoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = 0.001 # avoid grad explode

    def forward(self, pred_depth, gt_depth):
        """Forward function of loss.
        Args:
        Returns:
        """

        valid_mask = gt_depth > 0

        input = pred_depth[valid_mask]
        target = gt_depth[valid_mask]

        log_diff = torch.log(input + self.eps) - torch.log(target + self.eps)
        num_pixels = log_diff.new_tensor(len(log_diff))

        sirmse = torch.sqrt(torch.sum(torch.square(log_diff)) / num_pixels - torch.square(torch.sum(log_diff)) / torch.square(num_pixels))

        loss =  self.loss_weight * sirmse
        return loss