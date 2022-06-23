# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

@LOSSES.register_module()
class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, preds_S, preds_T, depth_gt_resized):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """

        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        loss = 0
        for i in range(N):
            mask_i = depth_gt_resized[i, :, :, :]
            feat_S = preds_S[i, :, :, :]
            feat_T = preds_T[i, :, :, :]

            valid_mask = mask_i > 0
            mask = valid_mask.expand(feat_S.shape).contiguous()
            valid_feat_S = feat_S[mask].reshape(C, -1)
            valid_feat_T = feat_T[mask].reshape(C, -1)

            softmax_feat_S = F.softmax(valid_feat_S / self.tau, dim=1)
            softmax_feat_T = F.softmax(valid_feat_T / self.tau, dim=1).detach()

            softmax_feat_S = softmax_feat_S.log()

            loss += self.criterion(softmax_feat_S, softmax_feat_T)

        loss = loss / (C * N)
        loss = self.loss_weight * (self.tau**2) * loss 

        return loss
