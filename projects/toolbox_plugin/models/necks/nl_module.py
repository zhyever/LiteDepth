# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from depth.models.builder import NECKS

import math
import torch

from mmcv.runner import BaseModule
from mmcv.cnn import NonLocal2d

@NECKS.register_module()
class NLNeck(BaseModule):
    """PPMNeck.

    PPMNeck

    Args:
        xxx
    """

    # dropout_ratio=0.1,
    def __init__(self,
                 channels=96,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',):
        super(NLNeck, self).__init__()

        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)
        
        # reduce
        # self.reduce = ConvModule(
        #     self.in_channels + self.channels * len(pool_scales),
        #     self.in_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg,
        # )

    # init weight
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # get the last-layer's output
        x = inputs[-1] 

        nl_out = self.nl_block(x)

        # replace the last-layer's output
        outs = []
        for feat in inputs[:-1]:
            outs.append(feat)
        outs.append(nl_out)

        return tuple(outs)
