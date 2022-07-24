# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES
import torch

@LOSSES.register_module()
class MarginLoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0, alpha=1, c=2, valid_mask=True, loss_type='l2'):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.c = c
        self.valid_mask = valid_mask
        self.loss_type = loss_type

    def forward(self,
                feat_s,
                feat_t,
                depth_gt_resized):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        if self.valid_mask:
            depth_gt = depth_gt_resized.expand(feat_s.size())
            mask = depth_gt > 0

            feat_s = feat_s[mask]
            feat_t = feat_t[mask]

        error = feat_s - feat_t

        if self.loss_type == 'l2':
            loss =  self.loss_weight * ((error)**2).mean()
        elif self.loss_type == 'robust':
            loss = self.loss_weight * (abs(self.alpha - 2) / self.alpha) * (torch.pow(torch.pow(error / self.c, 2)/abs(self.alpha - 2) + 1, self.alpha/2) - 1)
        elif self.loss_type == 'l1':
            loss =  self.loss_weight * (torch.abs(error)).mean()
        else:
            NotImplementedError

        return loss