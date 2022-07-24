# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np
import torch

from robust_loss_pytorch import AdaptiveLossFunction
import torch.distributed as dist

@LOSSES.register_module()
class RobustLoss(nn.Module):
    """RobustLoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0, alpha=1, c=2, log=False, adaptive=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.c = c

        self.eps = 0.001 # avoid grad explode
        self.log = log

        self.adaptive = adaptive
        if self.adaptive:
            self.adaptive_loss = AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device=dist.get_rank())


    def forward(self, pred_depth, gt_depth):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        mask = gt_depth > 0

        if self.log:
            gt_depth_masked = torch.log(gt_depth[mask] + self.eps)
            pred_depth_masked = torch.log(pred_depth[mask] + self.eps)
        else:
            gt_depth_masked = gt_depth[mask]
            pred_depth_masked = pred_depth[mask]

        error = gt_depth_masked - pred_depth_masked

        if self.adaptive:
            error = error.unsqueeze(dim=1)
            robust_loss = self.adaptive_loss.lossfun(error)

        else:
            robust_loss = (abs(self.alpha - 2) / self.alpha) * (torch.pow(torch.pow(error / self.c, 2)/abs(self.alpha - 2) + 1, self.alpha/2) - 1)

        loss =  self.loss_weight * robust_loss.mean()

        return loss