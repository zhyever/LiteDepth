# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES

@LOSSES.register_module()
class SimilarityMSELoss(nn.Module):
    """MSELoss.
    Args:
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
            feat_s (torch.Tensor): Feats from student
            feat_t (torch.Tensor): Feats form teacher
            depth_gt_resized (torch.Tensor): depth_gt_resized
        Returns:
            torch.Tensor: The calculated loss
        """
        N, C, H, W = feat_s.shape

        loss = 0
        for i in range(N):
            mask_i = depth_gt_resized[i, :, :, :]
            feat_S = feat_s[i, :, :, :]
            feat_T = feat_t[i, :, :, :]

            valid_mask = mask_i > 0
            mask = valid_mask.expand(feat_S.shape).contiguous()
            valid_feat_S = feat_S[mask].reshape(C, -1)
            valid_feat_T = feat_T[mask].reshape(C, -1)

            # norm the C dim
            valid_feat_S = F.normalize(valid_feat_S, p=2, dim=0)
            valid_feat_T = F.normalize(valid_feat_T, p=2, dim=0)

            similarity_S = valid_feat_S.permute(1, 0) @ valid_feat_S
            similarity_T = valid_feat_T.permute(1, 0) @ valid_feat_T

            loss += ((similarity_S - similarity_T)**2).mean()

        loss = self.loss_weight * loss / N

        return loss