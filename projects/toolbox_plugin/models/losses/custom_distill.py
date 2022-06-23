# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from depth.models.builder import LOSSES

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices

@LOSSES.register_module()
class CustomDistll(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        depth_max,
        depth_min,
        mode,
        num_bins,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(CustomDistll, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        self.criterion = nn.KLDivLoss(reduction='sum')

        self.depth_max = depth_max
        self.depth_min = depth_min
        self.mode = mode
        self.num_bins = num_bins
    

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

        depth_gt_indices = bin_depths(depth_gt_resized, self.mode, self.depth_min, self.depth_max, self.num_bins, target=True)

        proto_teacher = []
        proto_student = []
        for j in range(self.num_bins):
            range_mask = depth_gt_indices == j

            range_mask = range_mask.expand((N, C, H, W)).permute(1, 0, 2, 3).contiguous()

            feat_S_range = preds_S.permute(1, 0, 2, 3).contiguous()[range_mask].reshape(C, -1)
            feat_T_range = preds_T.permute(1, 0, 2, 3).contiguous()[range_mask].reshape(C, -1)

            if feat_S_range.numel() == 0 or feat_T_range.numel() == 0:
                continue
            
            feat_S_range_avg = F.adaptive_avg_pool1d(feat_S_range.unsqueeze(dim=0), 1).squeeze()
            feat_T_range_avg = F.adaptive_avg_pool1d(feat_T_range.unsqueeze(dim=0), 1).squeeze()

            # feat_S = F.softmax(feat_S_range_avg / self.tau, dim=0)
            # feat_T = F.softmax(feat_T_range_avg / self.tau, dim=0).detach()
            feat_S = F.normalize(feat_S_range_avg, p=2, dim=0)
            feat_T = F.normalize(feat_T_range_avg, p=2, dim=0)

            proto_student.append(feat_S)
            proto_teacher.append(feat_T)

        student_feats = torch.stack(proto_student, dim=1) # C, n_bins
        teacher_feats = torch.stack(proto_teacher, dim=1) # C, n_bins

        similarity_S = student_feats.permute(1, 0) @ student_feats # n_bins, n_bins
        similarity_T = teacher_feats.permute(1, 0) @ teacher_feats

        loss = self.loss_weight * ((similarity_S - similarity_T)**2).mean()

        return loss
