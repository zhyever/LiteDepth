# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np

@LOSSES.register_module()
class GradDepthErrorLoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, valid_mask=True, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.valid_mask = valid_mask

    def forward(self,
                pred_depth,
                gt_depth):
        """Forward function of loss.
        Args:
        Returns:
        """

        error_depth = gt_depth - pred_depth
        
        gt_depth = gt_depth.clone()
        if self.valid_mask:
            gt_depth[gt_depth == 0] = np.inf

        error_depth_x_grad = error_depth[:, :, :, 1:] - error_depth[:, :, :, :-1]
        error_depth_y_grad = error_depth[:, :, 1:, :] - error_depth[:, :, :-1, :]


        if self.valid_mask:

            error_depth_x_grad[torch.isnan(error_depth_x_grad)] = 0
            error_depth_x_grad[torch.isinf(error_depth_x_grad)] = 0
            error_depth_y_grad[torch.isnan(error_depth_y_grad)] = 0
            error_depth_y_grad[torch.isinf(error_depth_y_grad)] = 0

        loss = torch.mean(torch.abs(error_depth_x_grad)) + torch.mean(torch.abs(error_depth_y_grad))

        loss =  self.loss_weight * loss

        return loss