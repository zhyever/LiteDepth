# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from depth.ops import resize

from depth.models.builder import LOSSES

@LOSSES.register_module()
class SimilarityMSELoss(nn.Module):
    """MSELoss.
    Args:
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0, patch_w=4, patch_h=4):
        super().__init__()
        self.loss_weight = loss_weight

        self.patch_w = patch_w
        self.patch_h = patch_h
        self.maxpool = nn.MaxPool2d(kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w), padding=0, ceil_mode=True)

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

        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_s = self.maxpool(feat_s)
        feat_t= self.maxpool(feat_t)

        depth_gt_resized = resize(
            input=depth_gt_resized,
            size=[depth_gt_resized.shape[-2] // self.patch_h, depth_gt_resized.shape[-1] // self.patch_w],
            mode='nearest',
            align_corners=None,
            warning=False)

        loss = 0
        for i in range(N):
            mask_i = depth_gt_resized[i, :, :, :]
            feat_s_i = feat_s[i, :, :, :]
            feat_t_i = feat_t[i, :, :, :]

            valid_mask = mask_i > 0
            mask = valid_mask.expand(feat_s_i.shape).contiguous()
            valid_feat_s_i = feat_s_i[mask].reshape(C, -1)
            valid_feat_t_i = feat_t_i[mask].reshape(C, -1)

            # norm the C dim
            valid_feat_s_i = F.normalize(valid_feat_s_i, p=2, dim=0)
            valid_feat_t_i = F.normalize(valid_feat_t_i, p=2, dim=0)

            similarity_s_i = valid_feat_s_i.permute(1, 0) @ valid_feat_s_i
            similarity_t_i = valid_feat_t_i.permute(1, 0) @ valid_feat_t_i

            loss += ((similarity_s_i - similarity_t_i)**2).mean()

        loss = self.loss_weight * loss / N

        return loss