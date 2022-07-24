from depth.models.builder import DEPTHER
from depth.models.depther import BaseDepther
from depth.models import build_depther
import torch
import mmcls.models
import copy
import torch.nn as nn
from depth.ops import resize
from depth.models.builder import build_loss
from mmcv.runner import BaseModule, auto_fp16

@DEPTHER.register_module()
class DistillWrapper(BaseDepther):
    def __init__(self,
                 student_depther_cfg,
                 teacher_depther_cfg=None,
                 ema=False,
                 distill=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 teacher_select_de_index=(0, ),
                 student_select_de_index=(0, ),
                 distill_loss=None,
                 super_resolution=False,
                 upsample_type='nearest',
                 layer_weights=(1, ),
                 val_model='student',
                 pretrained=None,
                 fix_last=False):
        super(DistillWrapper, self).__init__(init_cfg)

        self.distill = distill
        self.teacher_depther_cfg = teacher_depther_cfg
        self.student_depther_cfg = student_depther_cfg
        self.student_depther = build_depther(student_depther_cfg)

        if self.distill:
            self.teacher_depther = build_depther(teacher_depther_cfg)
            self.freeze(self.teacher_depther)
            self.teacher_select_de_index = teacher_select_de_index
            self.student_select_de_index = student_select_de_index
            self.layer_weights = layer_weights
            self.distill_loss_cfg = distill_loss
            if isinstance(distill_loss, list):
                self.distill_loss = [build_loss(i) for i in distill_loss]
            else:
                self.distill_loss = build_loss(distill_loss)
            self.super_resolution = super_resolution # for teacher
            self.upsample_type = upsample_type

            if self.upsample_type == 'learned':
                self.projection_layer = nn.Sequential(
                    nn.PixelShuffle(2),
                    nn.Conv2d(student_depther_cfg.decode_head.channels//4, student_depther_cfg.decode_head.channels, 3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(student_depther_cfg.decode_head.channels//4, student_depther_cfg.decode_head.channels, 3, stride=1, padding=1),
                )
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.val_model = val_model
        
        self.ema = ema
        if self.ema:
            student_depther_cfg_copy = copy.deepcopy(student_depther_cfg)
            self.ema_model = build_depther(student_depther_cfg_copy)
            self.freeze(self.ema_model)
        
        self.fix_last = fix_last

    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(DistillWrapper, self).init_weights()

        if self.fix_last:
            self.student_depther.decode_head.conv_depth_1x1.weight = self.teacher_depther.decode_head.conv_depth_1x1.weight
            self.student_depther.decode_head.conv_depth_1x1.bias = self.teacher_depther.decode_head.conv_depth_1x1.bias
            self.freeze(self.student_depther.decode_head.conv_depth_1x1)

    def extract_feat(self, img_teacher, img_student, img_metas, depth_gt, **kwargs):
        """Placeholder for extract features from images."""

        # imgs_4x_student = resize(input=img_normed_student, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)

        if self.distill:
            # imgs_4x_teacher = resize(input=img_normed_teacher, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)

            if self.super_resolution:
                raise NotImplementedError
                # teacher_encoder_feats = self.teacher_depther.backbone(img_normed_teacher)
                # teacher_decoder_feats_act, teacher_decoder_feats, teacher_outputs, teacher_losses = \
                # self.teacher_depther.decode_head.forward_train(img_normed_teacher, 
                #                                                teacher_encoder_feats, 
                #                                                img_metas, 
                #                                                depth_gt, 
                #                                                self.train_cfg, 
                #                                                return_immediately=True, 
                #                                                **kwargs)

            else:
                teacher_encoder_feats = self.teacher_depther.backbone(img_teacher)

                teacher_decoder_feats_act, teacher_decoder_feats, teacher_outputs, teacher_losses = \
                self.teacher_depther.decode_head.forward_train(img_teacher, 
                                                               teacher_encoder_feats, 
                                                               img_metas, 
                                                               depth_gt, 
                                                               self.train_cfg, 
                                                               return_immediately=True, 
                                                               **kwargs)

        student_encoder_feats = self.student_depther.backbone(img_student)
        student_decoder_feats_act, student_decoder_feats, student_outputs, student_losses = \
            self.student_depther.decode_head.forward_train(img_student, 
                                                           student_encoder_feats, 
                                                           img_metas, 
                                                           depth_gt, 
                                                           self.train_cfg, 
                                                           return_immediately=True, 
                                                           **kwargs)

        if self.distill:
            return teacher_decoder_feats_act, teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
                student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses
        else:
            return student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses

    # for ema
    def extract_feat_ema(self, img_teacher, img_student, img_metas, depth_gt, **kwargs):
        # imgs_4x_student = resize(input=img_normed_student, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)

        student_encoder_feats = self.ema_model.backbone(img_student)
        _, _, _, ema_losses = \
            self.ema_model.decode_head.forward_train(img_student, 
                                                     student_encoder_feats, 
                                                     img_metas, 
                                                     depth_gt, 
                                                     self.train_cfg, 
                                                     return_immediately=True, 
                                                     **kwargs)

        return ema_losses


    def forward_train(self, imgs, img_metas, depth_gt, img_teacher, img_student, **kwargs):
        """Placeholder for Forward function for training."""

        if self.distill:
            teacher_decoder_feats_act, teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
                student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses = \
                    self.extract_feat(img_teacher, img_student, img_metas, depth_gt, **kwargs)

            total_losses = student_losses
            total_losses['decode.loss_depth'] = total_losses.pop('loss_depth')
            if 'loss_depth_grad' in total_losses.keys():
                total_losses['decode.loss_depth_grad'] = total_losses.pop('loss_depth_grad')

            # for i in teacher_decoder_feats:
            #     print(i.shape)
            # print("---")
            # for i in student_decoder_feats:
            #     print(i.shape)
            # exit(100)

            for idx, (idx_t, idx_s, w) in enumerate(zip(self.teacher_select_de_index, self.student_select_de_index, self.layer_weights)):
                teacher_last_layer_feat = teacher_decoder_feats[-(idx_t + 1)]
                student_last_layer_feat = student_decoder_feats[-(idx_s + 1)]

            
                # if self.upsample_type == 'bilinear':
                #     student_last_layer_feat = resize(
                #         input=student_last_layer_feat,
                #         size=teacher_last_layer_feat.shape[2:],
                #         mode='bilinear',
                #         align_corners=True,
                #         warning=False
                #     )
                # elif self.upsample_type == 'nearest':
                #     student_last_layer_feat = resize(
                #         input=student_last_layer_feat,
                #         size=teacher_last_layer_feat.shape[2:],
                #         mode='nearest',
                #         align_corners=None,
                #         warning=False
                #     )
                # elif self.upsample_type == 'learned':
                #     student_last_layer_feat = self.projection_layer(student_last_layer_feat)
                #     # raise NotImplementedError
                # else:
                #     raise NotImplementedError


                # depth_gt_resized = resize(
                #     input=depth_gt,
                #     size=teacher_last_layer_feat.shape[2:],
                #     mode='nearest',
                #     align_corners=None,
                #     warning=False
                # )
                
                depth_gt_resized = resize(
                    input=depth_gt,
                    size=student_last_layer_feat.shape[2:],
                    mode='bilinear',
                    align_corners=True,
                    warning=False
                )
                
                if isinstance(self.distill_loss, list):
                    distill_loss={}
                    for i in range(len(self.distill_loss)):
                        loss_temp = self.distill_loss[i](student_last_layer_feat, teacher_last_layer_feat, depth_gt_resized)
                        distill_loss['loss_distill_{}_{}'.format(self.distill_loss_cfg[i].type, idx_t)] = loss_temp * self.train_cfg.distill_loss_weight[i]
                else:
                    distill_loss = self.distill_loss(student_last_layer_feat, teacher_last_layer_feat, depth_gt_resized)
                    distill_loss = {'loss_distill_{}_{}'.format(self.distill_loss_cfg.type, idx_t): distill_loss * w}
                total_losses.update(**distill_loss)
                
            total_losses['depth_info_teacher'] = teacher_losses['loss_depth']
            if 'img_depth_pred' in teacher_losses.keys():
                total_losses['img_depth_pred_teacher'] = teacher_losses['img_depth_pred']
            

        else:
            student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses = self.extract_feat(imgs, img_metas, depth_gt, **kwargs)
            total_losses = student_losses
            

        if self.ema:
            ema_losses = self.extract_feat_ema(imgs, img_metas, depth_gt, **kwargs)
            total_losses['ema_info_depth'] = ema_losses['loss_depth'].detach()
        
        return total_losses
            

    # used in test, which is defined in depther model
    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass
        # return self.student_depther.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, img_teacher, img_student, **kwargs):
        """Placeholder for single image test."""

        if self.ema:
            return self.ema_model.simple_test(img_student, img_meta, **kwargs)
        if self.val_model == 'student':
            return self.student_depther.simple_test(img_student, img_meta, **kwargs)
        elif self.val_model == 'teacher':
            return self.teacher_depther.simple_test(img_teacher, img_meta, **kwargs)

        # return self.teacher_depther.simple_test(img, img_meta, **kwargs)
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs) 
    
    def forward_test(self, imgs, img_metas, img_teacher, img_student,**kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """

        return self.simple_test(imgs[0], img_metas[0], img_teacher[0], img_student[0], **kwargs)
