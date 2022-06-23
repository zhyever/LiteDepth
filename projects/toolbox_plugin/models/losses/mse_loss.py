# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                feat_s,
                feat_t,
                depth_gt_resized):
        """Forward function of loss.
        Args:
        Returns:
        """
        
        depth_gt = depth_gt_resized.expand(feat_s.size())
        mask = depth_gt > 0

        feat_s = feat_s[mask]
        feat_t = feat_t[mask]

        loss =  self.loss_weight * ((feat_s - feat_t)**2).mean()

        return loss