# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from depth.models.builder import NECKS

import torch

from mmcv.runner import BaseModule


@NECKS.register_module()
class PPMNeck(BaseModule):
    """PPMNeck.

    PPMNeck

    Args:
        xxx
    """

    # dropout_ratio=0.1,
    def __init__(self,
                 pool_scales=(1, 2, 3, 6), 
                 in_channels=96, 
                 channels=16, 
                 conv_cfg=None, 
                 norm_cfg=None, # dict(type='SyncBN', requires_grad=True) Offical
                 act_cfg=None, 
                 align_corners=True):
        super(PPMNeck, self).__init__()

        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.ppms = nn.ModuleList()
        for pool_scale in pool_scales:
            self.ppms.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))
        
        # reduce
        self.reduce = ConvModule(
            self.in_channels + self.channels * len(pool_scales),
            self.in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    # init weight
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            
    def psp_modules(self, x):
        ppm_outs = []
        for ppm in self.ppms:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

    def forward(self, inputs):
        # get the last-layer's output
        x = inputs[-1] 

        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)

        # reduce the feature dim to save resource
        psp_outs = self.reduce(psp_outs)

        # replace the last-layer's output
        outs = []
        for feat in inputs[:-1]:
            outs.append(feat)
        outs.append(psp_outs)
        return tuple(outs)
