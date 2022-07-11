from argparse import Action
import torch
from depth.models.builder import DEPTHER
from depth.ops import resize
from .encoder_decoder_mobile import DepthEncoderDecoderMobile
import copy
import torch.nn as nn


import torch.nn.functional as F
from timm.models.layers import pad_same
from timm.models.layers import Conv2dSame

def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_value=0):
    x = pad_same(x, weight.shape[-2:], stride, dilation, padding_value)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSameHackPadding(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, padding_value, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameHackPadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.padding_value = padding_value
    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_value)

@DEPTHER.register_module()
class DepthEncoderDecoderMobileTF(DepthEncoderDecoderMobile):
    r'''
    used convert pytorch model to the tflite
    '''

    def __init__(self,
                 downsample_target=(128, 160),
                 img_norm_cfg=dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                 **kwarg):
        super(DepthEncoderDecoderMobile, self).__init__(**kwarg)
        self.downsample_target = downsample_target

        self.img_norm_cfg = img_norm_cfg
        assert self.img_norm_cfg["mean"] == self.img_norm_cfg["std"], "only support mean=std now"
        self.img_norm_mean = torch.tensor(self.img_norm_cfg["mean"]).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.img_norm_std = torch.tensor(self.img_norm_cfg["std"]).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)

        self.decode_head.max_depth = self.decode_head.max_depth * 1000 # scale up 1000 for max_depth in sigmoid

    def forward(self, input):
        
        # norm the input image
        # input = (input - self.img_norm_mean.expand(input.shape)) / self.img_norm_std.expand(input.shape)
        # hack here
        input = (input - 127.5) / 127.5

        out = self.extract_feat(input)
        out = self.decode_head.forward(out, None)

        out = resize(
            input=out,
            size=(480, 640),
            mode='nearest',
            align_corners=None)
        
        return out

@DEPTHER.register_module()
class DepthEncoderDecoderMobileMergeTF(DepthEncoderDecoderMobile):
    r'''
    used convert pytorch model to the tflite
    '''

    def __init__(self,
                 downsample_target=(128, 160),
                 img_norm_cfg=dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]), # merge image norm
                 **kwarg):
        super(DepthEncoderDecoderMobileMergeTF, self).__init__(**kwarg)

        self.img_norm_cfg = img_norm_cfg
        assert self.img_norm_cfg["mean"] == self.img_norm_cfg["std"], "only support mean = std now"
        self.img_norm_mean = torch.tensor(self.img_norm_cfg["mean"])
        self.img_norm_std = torch.tensor(self.img_norm_cfg["std"])

        self.downsample_target = downsample_target
        self.decode_head.max_depth = self.decode_head.max_depth * 1000 # scale up 1000 for max_depth in sigmoid


    def merge_image_normalization(self):
        
        template = copy.deepcopy(self.backbone.timm_model.conv_stem)

        # ensure the same input padding (only work when img_norm.mean = img_norm.std)
        padding_value = 0 * self.img_norm_std[0] + self.img_norm_mean[0]
        first_conv = Conv2dSameHackPadding(
            padding_value, 
            template.in_channels, 
            template.out_channels, 
            template.kernel_size, 
            stride=template.stride,
            padding=template.padding, 
            dilation=template.dilation, 
            groups=template.groups, 
            bias=template.bias)

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

        # reload
        first_conv.weight = nn.Parameter(first_conv_weight_new)
        first_conv.bias = nn.Parameter(bias_tensor)

        # replace
        self.backbone.timm_model.conv_stem = first_conv

    def extract_feat(self, img):
        """Extract features from images."""

        img = resize(input=img, 
                     size=(self.downsample_target[0], self.downsample_target[1]), 
                     mode='bilinear', 
                     align_corners=True)

        x = self.backbone(img)
        return x
    
    def forward(self, input):
        
        out = self.extract_feat(input)
        out = self.decode_head.forward(out, None)

        out = resize(
            input=out,
            size=(480, 640),
            mode='nearest',
            align_corners=None)
        
        return out
