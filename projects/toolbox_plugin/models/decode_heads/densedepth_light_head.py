from distutils.command.build import build
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

from projects.toolbox_plugin.models.utils.dbb_block import DiverseBranchBlock

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    Reduce to only one 3x3 conv
    
    '''
    def __init__(self, up_features, skip_features, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None, final_conv=False):
        super(UpSample, self).__init__()

        self.final_conv = final_conv

        if final_conv is not True:
            self.convA = ConvModule(up_features+skip_features, output_features, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.hack_act = build_activation_layer(act_cfg)

    def forward(self, x, concat_with, return_immediately=False):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)

        if self.final_conv:    
            temp = torch.cat([up_x, concat_with], dim=1)
            if return_immediately:
                return temp, temp
            else:
                return temp

        else:
            if return_immediately:
                temp = self.convA(torch.cat([up_x, concat_with], dim=1), activate=False)
                out = self.hack_act(temp)
                return out, temp
            else:
                return self.convA(torch.cat([up_x, concat_with], dim=1))

@HEADS.register_module()
class DenseDepthHeadLightMobile(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
    """

    def __init__(self,
                 up_sample_channels,
                 debug=False,
                 with_loss_depth_grad=False,
                 loss_depth_grad=None,
                 with_loss_ssim=False,
                 loss_ssim=None,
                 extend_up_conv_num=0,
                 upsample_type='nearest',
                 in_index=(0,1,2,3,4),
                 logits_dim=0,
                 with_loss_vnl=False,
                 loss_vnl=None,
                 with_loss_pair=False,
                 loss_pair=None,
                 with_loss_robust=False,
                 loss_robust=None,
                 dbb_block=False,
                 with_loss_auto_weight=False,
                 loss_auto_weight=None,
                 with_loss_sirmse=False,
                 loss_sirmse=None,
                 with_loss_grad_error=False,
                 loss_grad_error=None,
                 **kwargs):
        super(DenseDepthHeadLightMobile, self).__init__(**kwargs)

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
                        act_cfg=self.act_cfg
                    ))
            elif index == len(self.in_channels) - 1:
                self.conv_list.append(
                    UpSample(up_features=up_channel_temp,
                             skip_features=in_channel,
                             output_features=up_channel,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg,
                             final_conv=True))
            else:
                self.conv_list.append(
                    UpSample(up_features=up_channel_temp,
                             skip_features=in_channel,
                             output_features=up_channel,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg))

            # save earlier fusion target
            up_channel_temp = up_channel
        
        self.with_loss_depth_grad = with_loss_depth_grad
        if self.with_loss_depth_grad:
            self.loss_depth_grad = build_loss(loss_depth_grad)
        
        self.with_loss_ssim = with_loss_ssim
        if self.with_loss_ssim:
            self.loss_ssim = build_loss(loss_ssim)
        
        self.with_loss_vnl = with_loss_vnl
        if self.with_loss_vnl:
            self.loss_vnl = build_loss(loss_vnl)
        
        self.with_loss_pair = with_loss_pair
        if self.with_loss_pair:
            self.loss_pair = build_loss(loss_pair)

        self.with_loss_robust = with_loss_robust
        if self.with_loss_robust:
            self.loss_robust = build_loss(loss_robust)
        
        self.with_loss_auto_weight = with_loss_auto_weight
        if self.with_loss_auto_weight:
            self.loss_auto_weight = build_loss(loss_auto_weight)

        
        self.with_loss_sirmse = with_loss_sirmse
        if self.with_loss_sirmse:
            self.loss_sirmse = build_loss(loss_sirmse)
        
        self.with_loss_grad_error = with_loss_grad_error
        if self.with_loss_grad_error:
            self.loss_grad_error = build_loss(loss_grad_error)

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

        self.upsample_type = upsample_type
        self.debug = debug

        self.in_index = in_index
        self.logits_dim = logits_dim
        final_input_dim = self.in_channels[::-1][0] + self.up_sample_channels[::-1][1]

        self.dbb_block = dbb_block

        if self.dbb_block:
            self.conv_depth_3x3 = DiverseBranchBlock(final_input_dim, self.logits_dim, kernel_size=3, padding=1, stride=1, deploy=False)
        else:
            self.conv_depth_3x3 = nn.Conv2d(final_input_dim, self.logits_dim, kernel_size=3, padding=1, stride=1)
            
        self.conv_depth_1x1 = nn.Conv2d(self.logits_dim, 1, kernel_size=1, padding=0, stride=1)
        self.conv_depth_act = nn.ReLU()
        
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
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
        inputs = [inputs[i] for i in self.in_index]

        outputs = {}

        if return_immediately:
            temp_feat_list = []
            temp_feat_before_act_list = []
            
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                    temp_feat_before_act_list.append(temp_feat)
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
            
            temp = self.conv_depth_3x3(temp_feat_list[-1])
            temp_feat_before_act_list.append(temp)
            temp = self.conv_depth_act(temp)
            depth_pred = self.conv_depth_act(self.conv_depth_1x1(temp))

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
            
            temp = self.conv_depth_3x3(temp_feat_list[-1])
            temp = self.conv_depth_act(temp)
            depth_pred = self.conv_depth_act(self.conv_depth_1x1(temp))


        outputs['depth_pred'] = depth_pred

        losses = self.losses(outputs, depth_gt, img_metas)

        if self.debug:
            log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
            losses.update(**log_imgs)

        if return_immediately:
            return temp_feat_list, temp_feat_before_act_list, outputs, losses
        
        else:
            return losses

    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, 
               model_outputs, 
               depth_gt,
               img_metas):
        """Compute depth loss."""
        loss = dict()

        depth_pred = model_outputs['depth_pred']
        
        if self.upsample_type == 'nearest':
            depth_pred_upsample = resize(
                input=depth_pred,
                size=depth_gt.shape[2:],
                mode='nearest',
                align_corners=None)

        elif self.upsample_type == 'bilinear':
            depth_pred_upsample = resize(
                input=depth_pred,
                size=depth_gt.shape[2:],
                mode='bilinear',
                align_corners=True,
                warning=False)

        depth_gt_downsample = resize(
            input=depth_gt,
            size=depth_pred.shape[2:],
            mode='bilinear',
            align_corners=True,
            warning=False)

        # NOTE: a simple hack version
        if self.with_loss_auto_weight:
            
            loss_depth = self.loss_decode(depth_pred_upsample, depth_gt)
            loss_depth_info = loss_depth.clone().detach()
            loss['info_sigloss'] = loss_depth_info

            depth_grad, gt_x_grad, gt_y_grad, pred_x_grad, pred_y_grad = self.loss_depth_grad(depth_pred, depth_gt_downsample, debug=True)
            loss_depth_grad = depth_grad

            loss_depth_vnl = self.loss_vnl(depth_pred, depth_gt_downsample)
            
            loss_depth_robust_loss = self.loss_robust(depth_pred_upsample, depth_gt)

            loss['loss_depth'] = self.loss_auto_weight(loss_depth, loss_depth_grad, loss_depth_vnl, loss_depth_robust_loss)

            weight_info = self.loss_auto_weight.params.clone().detach()
            loss['weight_sigloss'] = weight_info[0]
            loss['weight_grad'] = weight_info[1]
            loss['weight_vnl'] = weight_info[2]
            loss['weight_robust'] = weight_info[3]

        else:

            loss['loss_depth'] = self.loss_decode(depth_pred_upsample, depth_gt)

            if self.with_loss_depth_grad:
                # generate depth grad

                if self.debug:
                    depth_grad, gt_x_grad, gt_y_grad, pred_x_grad, pred_y_grad = self.loss_depth_grad(depth_pred, depth_gt_downsample, debug=True)
                    loss["img_gt_x_grad"] = gt_x_grad[0]
                    loss["img_gt_y_grad"] = gt_y_grad[0]
                    loss["img_pred_x_grad"] = pred_x_grad[0]
                    loss["img_pred_y_grad"] = pred_y_grad[0]
                    loss['loss_depth_grad'] = depth_grad
                else:
                    depth_grad = self.loss_depth_grad(depth_pred, depth_gt_downsample)
                    loss['loss_depth_grad'] = depth_grad

            if self.with_loss_ssim:
                loss['loss_depth_ssim'] = self.loss_ssim(depth_pred_upsample, depth_gt)
            
            if self.with_loss_vnl:
                loss['loss_depth_vnl'] = self.loss_vnl(depth_pred, depth_gt_downsample)
            
            if self.with_loss_pair:
                loss['loss_depth_pair_loss'] = self.loss_pair(depth_pred, depth_gt_downsample)

            if self.with_loss_robust:
                loss['loss_depth_robust_loss'] = self.loss_robust(depth_pred_upsample, depth_gt)
            
            if self.with_loss_sirmse:
                loss['loss_depth_sirmse_loss'] = self.loss_sirmse(depth_pred_upsample, depth_gt)

            if self.with_loss_grad_error:
                loss['loss_grad_error'] = self.loss_grad_error(depth_pred_upsample, depth_gt)

        return loss

    # only for test. no aux branch used here.
    def forward(self, inputs, img_metas):
        """Forward function."""

        inputs = [inputs[i] for i in self.in_index]
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

        temp = self.conv_depth_3x3(temp_feat_list[-1])
        temp = self.conv_depth_act(temp)
        output = self.conv_depth_act(self.conv_depth_1x1(temp))

        return output
