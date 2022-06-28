# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.
    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss


@LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        # assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                preds_S,
                preds_T,
                depth_gt_resized,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        
        mask = depth_gt_resized.contiguous().view(-1) > 0

        s_feats = preds_S.permute(0, 2, 3, 1)
        s_feats = s_feats.contiguous().view(-1, s_feats.shape[-1])
        s_feats = s_feats[mask, :]

        t_feats = preds_T.permute(0, 2, 3, 1)
        t_feats = t_feats.contiguous().view(-1, t_feats.shape[-1])
        t_feats = t_feats[mask, :]

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            s_feats,
            t_feats,
            weight=None,
            reduction=self.reduction,
            avg_factor=None,
            T=self.T)

        return loss_kd