# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np

@LOSSES.register_module()
class GradDepthLoss(nn.Module):
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
                gt_depth,
                debug=False):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        if self.valid_mask:
            gt_depth[gt_depth == 0] = np.inf

        pred_x_grad = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
        pred_y_grad = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]

        gt_x_grad = gt_depth[:, :, :, 1:] - gt_depth[:, :, :, :-1]
        gt_y_grad = gt_depth[:, :, 1:, :] - gt_depth[:, :, :-1, :]

        if self.valid_mask:

            gt_x_grad[torch.isnan(gt_x_grad)] = 0
            gt_x_grad[torch.isinf(gt_x_grad)] = 0
            gt_y_grad[torch.isnan(gt_y_grad)] = 0
            gt_y_grad[torch.isinf(gt_y_grad)] = 0

            pred_x_grad[torch.isnan(gt_x_grad)] = 0
            pred_x_grad[torch.isinf(gt_x_grad)] = 0
            pred_y_grad[torch.isnan(gt_y_grad)] = 0
            pred_y_grad[torch.isinf(gt_y_grad)] = 0

        loss_x = torch.mean(torch.abs(pred_x_grad - gt_x_grad))
        loss_y = torch.mean(torch.abs(pred_y_grad - gt_y_grad))

        loss =  self.loss_weight * (loss_x + loss_y)

        if debug:
            return loss, torch.abs(gt_x_grad), torch.abs(gt_y_grad), torch.abs(pred_x_grad), torch.abs(pred_y_grad)
        
        else:
            return loss