import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
class StudentSegContrast(nn.Module):
    def __init__(self, num_classes, pixel_memory_size, region_memory_size, region_contrast_size, pixel_contrast_size, 
                 contrast_kd_temperature, contrast_temperature, ignore_label, loss_weight, pixel_weight, region_weight, 
                 dim, depth_min=1e-3, depth_max=40, mode='LID', downsample=1):
        super(StudentSegContrast, self).__init__()
        self.base_temperature = 0.1
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.dim = dim
        self.ignore_label = ignore_label

        self.num_classes = num_classes
        self.region_memory_size = region_memory_size
        self.pixel_memory_size = pixel_memory_size
        self.pixel_update_freq = 16
        self.pixel_contrast_size = pixel_contrast_size
        self.region_contrast_size = region_contrast_size


        self.register_buffer("teacher_segment_queue", torch.randn(self.num_classes, self.region_memory_size, self.dim))
        self.teacher_segment_queue = nn.functional.normalize(self.teacher_segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("teacher_pixel_queue", torch.randn(self.num_classes, self.pixel_memory_size, self.dim))
        self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.mode = mode
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self.region_weight = region_weight
        self.downsample = downsample


    def _sample_negative(self, Q, index):
        class_num, cache_size, feat_size = Q.shape
        contrast_size = index.size(0)
        X_ = torch.zeros((class_num * contrast_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * contrast_size, 1)).float().cuda()
        sample_ptr = 0

        
        for ii in range(class_num):
            this_q = Q[ii, index, :]
            X_[sample_ptr:sample_ptr + contrast_size, ...] = this_q
            y_[sample_ptr:sample_ptr + contrast_size, ...] = ii
            sample_ptr += contrast_size

        return X_, y_


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    
    def _dequeue_and_enqueue(self, keys, labels):
        segment_queue = self.teacher_segment_queue
        pixel_queue = self.teacher_pixel_queue

        keys = self.concat_all_gather(keys)
        labels = self.concat_all_gather(labels)
        
        batch_size, feat_dim, H, W = keys.size()

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.region_memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)    
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.pixel_memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + K) % self.pixel_memory_size


    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits/self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature**2
        return sim_dis


    def forward(self, s_feats, t_feats, depth_gt_resized):
        if self.downsample != 1:
            depth_gt_resized = F.interpolate(
                depth_gt_resized, 
                size=[depth_gt_resized.size(2)//self.downsample, depth_gt_resized.size(3)//self.downsample], 
                mode='nearest', 
                align_corners=None)

            s_feats = F.interpolate(
                s_feats, 
                size=[s_feats.size(2)//self.downsample, s_feats.size(3)//self.downsample], 
                mode='bilinear', 
                align_corners=True)
            
            t_feats = F.interpolate(
                t_feats, 
                size=[t_feats.size(2)//self.downsample, t_feats.size(3)//self.downsample], 
                mode='bilinear', 
                align_corners=True)

        labels = bin_depths(depth_gt_resized, self.mode, self.depth_min, self.depth_max, self.num_classes, target=True)

        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == s_feats.shape[-1], '{} {}'.format(labels.shape, s_feats.shape)

        ori_s_fea = s_feats
        ori_t_fea = t_feats
        ori_labels = labels

        batch_size = s_feats.shape[0]

        labels = labels.contiguous().view(-1)

        idxs = (labels != self.ignore_label)
        s_feats = s_feats.permute(0, 2, 3, 1)
        s_feats = s_feats.contiguous().view(-1, s_feats.shape[-1])
        s_feats = s_feats[idxs, :]

        t_feats = t_feats.permute(0, 2, 3, 1)
        t_feats = t_feats.contiguous().view(-1, t_feats.shape[-1])
        t_feats = t_feats[idxs, :]

        self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone())

        if idxs.sum() == 0: # just a trick to skip the case of having no semantic embeddings
            return 0. * (s_feats**2).mean(), 0. * (s_feats**2).mean()
            
        class_num, pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        perm = torch.randperm(pixel_queue_size)
        pixel_index = perm[:self.pixel_contrast_size]
        t_X_pixel_contrast, _ = self._sample_negative(self.teacher_pixel_queue, pixel_index)


        t_pixel_logits = torch.div(torch.mm(t_feats, t_X_pixel_contrast.T), self.contrast_temperature)
        s_pixel_logits = torch.div(torch.mm(s_feats, t_X_pixel_contrast.T), self.contrast_temperature)


        class_num, region_queue_size, feat_size = self.teacher_segment_queue.shape
        perm = torch.randperm(region_queue_size)
        region_index = perm[:self.region_contrast_size]
        t_X_region_contrast, _ = self._sample_negative(self.teacher_segment_queue, region_index)

        
        t_region_logits = torch.div(torch.mm(t_feats, t_X_region_contrast.T), self.contrast_temperature)
        s_region_logits = torch.div(torch.mm(s_feats, t_X_region_contrast.T), self.contrast_temperature)

        pixel_sim_dis = self.contrast_sim_kd(s_pixel_logits, t_pixel_logits.detach())
        region_sim_dis = self.contrast_sim_kd(s_region_logits, t_region_logits.detach())
        
        return self.loss_weight * (pixel_sim_dis * self.pixel_weight + region_sim_dis * self.pixel_weight)


