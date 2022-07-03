import copy
import torch
import torch.nn as nn

from depth.models.depther import DepthEncoderDecoder
from depth.models.builder import DEPTHER
from depth.ops import resize

import torch.nn.functional as F
from timm.models.layers import pad_same

from mmcv.utils import print_log
from depth.utils import get_root_logger

def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_value=0):
    x = pad_same(x, weight.shape[-2:], stride, dilation, padding_value)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, padding_value, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.padding_value = padding_value
    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_value)


@DEPTHER.register_module()
class DepthEncoderDecoderMobileMerge(DepthEncoderDecoder):
    r'''
    used in mobileAI challenge
    '''

    def __init__(self,
                 downsample_ratio=4,
                 img_norm_cfg=dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                 **kwarg):
        super(DepthEncoderDecoderMobileMerge, self).__init__(**kwarg)

        self.downsample_ratio = downsample_ratio
        self.img_norm_cfg = img_norm_cfg
        assert self.img_norm_cfg["mean"] == self.img_norm_cfg["std"], "only support mean==std now"
        self.img_norm_mean = torch.tensor(self.img_norm_cfg["mean"])
        self.img_norm_std = torch.tensor(self.img_norm_cfg["std"])

    def init_weights(self):
        super(DepthEncoderDecoderMobileMerge, self).init_weights()
        print_log(f'Start to merge image normalization into the first pre-trained Conv layer', logger=get_root_logger())
        self.merge_image_normalization()
        print_log(f'Successfully merge image normalization into the first pre-trained Conv layer', logger=get_root_logger())

    def merge_image_normalization(self):
        
        template = copy.deepcopy(self.backbone.timm_model.conv_stem)

        padding_value = 0 * self.img_norm_std[0] + self.img_norm_mean[0]

        first_conv = Conv2dSame(
            padding_value, 
            template.in_channels, 
            template.out_channels, 
            template.kernel_size, 
            stride=template.stride,
            padding=template.padding, 
            dilation=template.dilation, 
            groups=template.groups, 
            bias=template.bias)

        # first_conv = copy.deepcopy(self.backbone.timm_model.conv_stem)

        # update weight
        first_conv_weight = self.backbone.timm_model.conv_stem._parameters['weight']
        target_dim, input_dim, h, w = first_conv_weight.shape
        mean = self.img_norm_mean.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand((target_dim, 3, h, w))
        std = self.img_norm_std.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand((target_dim, 3, h, w))
        first_conv_weight_new = first_conv_weight / std

        # update bias
        bias_list = []
        for i in range(target_dim):
            filter = first_conv_weight[i, :, :, :]
            bias = - torch.sum((mean/std)[i, :, :, :] * filter)
            bias_list.append(bias)
        bias_tensor = torch.stack(bias_list)

        first_conv.weight = nn.Parameter(first_conv_weight_new)
        first_conv.bias = nn.Parameter(bias_tensor)

        self.backbone.timm_model.conv_stem = first_conv

    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        # crop the pred depth to the certain range.
        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        if rescale:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='nearest')
        return out

    def extract_feat(self, img):
        """Extract features from images."""

        # x4 downsample the input image for speed up
        img = resize(input=img, 
                     size=(img.shape[-2] // self.downsample_ratio, img.shape[-1] // self.downsample_ratio), 
                     mode='bilinear', 
                     align_corners=True)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
    
        return x
