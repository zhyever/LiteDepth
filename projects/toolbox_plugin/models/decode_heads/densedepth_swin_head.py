import torch
import copy
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from depth.models.builder import HEADS, build_loss
import torch.nn.functional as F

from depth.models.decode_heads import DepthBaseDecodeHead
from depth.ops import resize
from mmcv.runner import force_fp32

from mmcv.cnn.bricks.activation import build_activation_layer

from mmcv.cnn import ConvModule, xavier_init

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    For swin teacher model (effective)
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.hack_act = build_activation_layer(act_cfg)

    def forward(self, x, concat_with, return_immediately=False):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        if return_immediately:
            temp = self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)), activate=False)
            out = self.hack_act(temp)
            return out, temp
        else:
            return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))

@HEADS.register_module()
class DenseDepthHeadSwinMobile(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
    """

    def __init__(self,
                 up_sample_channels,
                 with_depth_grad=False,
                 loss_depth_grad=None,
                 extend_up_conv_num=0,
                 **kwargs):
        super(DenseDepthHeadSwinMobile, self).__init__(**kwargs)

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
        
        self.with_depth_grad = with_depth_grad
        if self.with_depth_grad:
            self.loss_depth_grad = build_loss(loss_depth_grad)

        self.extend_convs = nn.ModuleList()
        self.extend_up_conv_num = extend_up_conv_num
        for i in range(self.extend_up_conv_num):
            self.extend_convs.append(
                ConvModule(
                    in_channels=up_sample_channels[0],
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        self.hack_act = build_activation_layer(self.act_cfg)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward_train(self, 
                      img, 
                      inputs, 
                      img_metas, 
                      depth_gt, 
                      train_cfg,
                      return_immediately=False):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # for i in inputs:
        #     print(i.shape)

        outputs = {}

        if return_immediately:
            temp_feat_list = []
            temp_feat_before_act_list = []
            
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat, temp_feat_before_act = self.conv_list[index](up_feat, skip_feat, return_immediately=True)
                    temp_feat_list.append(temp_feat)
                    temp_feat_before_act_list.append(temp_feat_before_act)

            for i in range(self.extend_up_conv_num):
                temp_feat = F.interpolate(
                    temp_feat_list[-1], 
                    size=[temp_feat_list[-1].size(2)*2, temp_feat_list[-1].size(3)*2], 
                    mode='bilinear', 
                    align_corners=True)
                temp_feat_before_act = self.extend_convs[i](temp_feat, activate=False)
                temp_feat = self.hack_act(temp_feat_before_act)
                temp_feat_list.append(temp_feat)
                temp_feat_before_act_list.append(temp_feat_before_act)
                
            depth_pred = self.depth_pred(temp_feat_list[-1])

        else:
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
                    
            for i in range(self.extend_up_conv_num):
                temp_feat = F.interpolate(
                    temp_feat_list[-1], 
                    size=[temp_feat_list[-1].size(2)*2, temp_feat_list[-1].size(3)*2], 
                    mode='bilinear', 
                    align_corners=True)
                temp_feat = self.extend_convs[i](temp_feat)
                temp_feat_list.append(temp_feat)
            depth_pred = self.depth_pred(temp_feat_list[-1])


        outputs['depth_pred'] = depth_pred

        losses = self.losses(outputs, depth_gt)

        if return_immediately:
            return temp_feat_list, temp_feat_before_act_list, outputs, losses
        
        else:
            return losses

    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, 
               model_outputs, 
               depth_gt):
        """Compute depth loss."""
        loss = dict()

        resized_output = {}

        for k,v in model_outputs.items():
            resized_v = resize(
                input=v,
                size=depth_gt.shape[2:],
                mode='nearest',
                align_corners=None)
            resized_output[k] = resized_v
        
        loss['loss_depth'] = self.loss_decode(resized_output['depth_pred'], depth_gt)

        if self.with_depth_grad:
            # generate depth grad
            valid_mask = depth_gt > 0
            valid_mask_x = valid_mask[:, :, :, :-1]
            valid_mask_y = valid_mask[:, :, :-1, :]

            x_grad_pred = resized_output['depth_pred'][:, :, :, 1:] - resized_output['depth_pred'][:, :, :, :-1]
            y_grad_pred = resized_output['depth_pred'][:, :, 1:, :] - resized_output['depth_pred'][:, :, :-1, :]
            x_grad_gt = depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1]
            y_grad_gt = depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :]

            depth_grad = self.loss_depth_grad(x_grad_pred[valid_mask_x], x_grad_gt[valid_mask_x]) +\
                self.loss_depth_grad(y_grad_pred[valid_mask_y], y_grad_gt[valid_mask_y])

            loss['loss_depth_grad'] = depth_grad

        return loss

    # only for test. no aux branch used here.
    def forward(self, inputs, img_metas):
        """Forward function."""

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

        for i in range(self.extend_up_conv_num):
            temp_feat = F.interpolate(
                temp_feat_list[-1], 
                size=[temp_feat_list[-1].size(2)*2, temp_feat_list[-1].size(3)*2], 
                mode='bilinear', 
                align_corners=True)
            temp_feat = self.extend_convs[i](temp_feat)
            temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])

        return output
