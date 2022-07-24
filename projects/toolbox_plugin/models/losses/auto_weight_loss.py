# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from depth.models.builder import LOSSES

@LOSSES.register_module()
class AutoReweightLoss(nn.Module):
    """AutoReweightLoss.
    Args:
    """

    def __init__(self, num=2):
        super(AutoReweightLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum