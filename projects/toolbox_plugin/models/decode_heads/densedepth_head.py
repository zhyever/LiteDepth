from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding
from torch.nn.modules import conv

from depth.models.builder import HEADS
import torch.nn.functional as F
from depth.models.utils import UpConvBlock, BasicConvBlock

from depth.models.decode_heads import DepthBaseDecodeHead
from depth.models.builder import DEPTHER
from depth.ops import resize

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))

@HEADS.register_module()
class DenseDepthHeadMobile(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
    """

    def __init__(self,
                 up_sample_channels,
                 **kwargs):
        super(DenseDepthHeadMobile, self).__init__(**kwargs)

        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = self.in_channels[::-1]

        # construct the decoder
        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(
                zip(self.in_channels, self.up_sample_channels)):
            if index == 0:
                self.conv_list.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=up_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        act_cfg=None
                    ))
            else:
                self.conv_list.append(
                    UpSample(skip_input=in_channel + up_channel_temp,
                                output_features=up_channel,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg))

            # save earlier fusion target
            up_channel_temp = up_channel

    def forward(self, inputs, img_metas):
        """Forward function."""
        
        # torch.Size([16, 16, 208, 272])
        # torch.Size([16, 24, 104, 136])
        # torch.Size([16, 40, 52, 68])
        # torch.Size([16, 112, 26, 34])
        # torch.Size([16, 320, 13, 17])
        # torch.Size([16, 1, 208, 272])

        # print(len(inputs))
        # for i in inputs:
        #     print(i.shape)

        temp_feat_list = []
        
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])

        return output
