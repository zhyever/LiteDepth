# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

import numpy as np

@LOSSES.register_module()
class SSIMDepthLoss(nn.Module):
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

        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self,
                pred_depth,
                gt_depth,
                debug=False):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        if self.valid_mask:
            valid_mask = gt_depth > 0


        x = self.refl(pred_depth)
        y = self.refl(gt_depth)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        
        ssim = torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

        if self.valid_mask:
            ssim_valid = ssim[valid_mask]

        loss =  self.loss_weight * torch.mean(ssim_valid)

        return loss