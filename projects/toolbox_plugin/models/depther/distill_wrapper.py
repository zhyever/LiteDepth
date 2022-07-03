from depth.models.builder import DEPTHER
from depth.models.depther import BaseDepther
from depth.models import build_depther
import torch
import mmcls.models
import copy
import torch.nn as nn
from depth.ops import resize
from depth.models.builder import build_loss

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
                 img_norm_cfg_teacher=None,
                 img_norm_cfg_student=None,
                 val_model='student',
                 pretrained=None):
        super(DistillWrapper, self).__init__(init_cfg)

        self.distill = distill
        self.teacher_depther_cfg = teacher_depther_cfg
        self.student_depther_cfg = student_depther_cfg
        self.student_depther = build_depther(student_depther_cfg)
        self.img_norm_cfg_student = img_norm_cfg_student

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
            self.img_norm_cfg_teacher = img_norm_cfg_teacher
            self.super_resolution = super_resolution # for teacher
            self.upsample_type = upsample_type
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.val_model = val_model
        
        self.ema = ema
        if self.ema:
            student_depther_cfg_copy = copy.deepcopy(student_depther_cfg)
            self.ema_model = build_depther(student_depther_cfg_copy)
            self.freeze(self.ema_model)

    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(DistillWrapper, self).init_weights()

    def extract_feat(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for extract features from images."""
        
        mean_student = imgs.new_tensor(self.img_norm_cfg_student.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std_student = imgs.new_tensor(self.img_norm_cfg_student.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        img_normed_student = (imgs - mean_student) / std_student
        imgs_4x_student = resize(input=img_normed_student, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)

        if self.distill:
            mean_teacher = imgs.new_tensor(self.img_norm_cfg_teacher.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            std_teacher = imgs.new_tensor(self.img_norm_cfg_teacher.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            img_normed_teacher = (imgs - mean_teacher) / std_teacher

            imgs_4x_teacher = resize(input=img_normed_teacher, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)


            if self.super_resolution:
                teacher_encoder_feats = self.teacher_depther.backbone(img_normed_teacher)

                teacher_decoder_feats_act, teacher_decoder_feats, teacher_outputs, teacher_losses = \
                self.teacher_depther.decode_head.forward_train(img_normed_teacher, 
                                                            teacher_encoder_feats, 
                                                            img_metas, 
                                                            depth_gt, 
                                                            self.train_cfg, 
                                                            return_immediately=True, 
                                                            **kwargs)
            else:
                teacher_encoder_feats = self.teacher_depther.backbone(imgs_4x_teacher)

                teacher_decoder_feats_act, teacher_decoder_feats, teacher_outputs, teacher_losses = \
                self.teacher_depther.decode_head.forward_train(imgs_4x_teacher, 
                                                            teacher_encoder_feats, 
                                                            img_metas, 
                                                            depth_gt, 
                                                            self.train_cfg, 
                                                            return_immediately=True, 
                                                            **kwargs)

        student_encoder_feats = self.student_depther.backbone(imgs_4x_student)
        student_decoder_feats_act, student_decoder_feats, student_outputs, student_losses = \
            self.student_depther.decode_head.forward_train(imgs_4x_student, 
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
    def extract_feat_ema(self, imgs, img_metas, depth_gt, **kwargs):

        mean_student = imgs.new_tensor(self.img_norm_cfg_student.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std_student = imgs.new_tensor(self.img_norm_cfg_student.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        img_normed_student = (imgs - mean_student) / std_student
        imgs_4x_student = resize(input=img_normed_student, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=True)

        student_encoder_feats = self.ema_model.backbone(imgs_4x_student)
        _, _, _, ema_losses = \
            self.ema_model.decode_head.forward_train(imgs_4x_student, 
                                                     student_encoder_feats, 
                                                     img_metas, 
                                                     depth_gt, 
                                                     self.train_cfg, 
                                                     return_immediately=True, 
                                                     **kwargs)

        return ema_losses


    def forward_train(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for Forward function for training."""

        if self.distill:
            teacher_decoder_feats_act, teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
                student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses = self.extract_feat(imgs, img_metas, depth_gt, **kwargs)

            student_losses['decode.loss_depth'] = student_losses.pop('loss_depth')

            for idx, (idx_t, idx_s, w) in enumerate(zip(self.teacher_select_de_index, self.student_select_de_index, self.layer_weights)):
                teacher_last_layer_feat = teacher_decoder_feats[-(idx_t + 1)]
                student_last_layer_feat = student_decoder_feats[-(idx_s + 1)]

                if self.upsample_type == 'bilinear':
                    student_last_layer_feat = resize(
                        input=student_last_layer_feat,
                        size=teacher_last_layer_feat.shape[2:],
                        mode='bilinear',
                        align_corners=True,
                        warning=False
                    )
                elif self.upsample_type == 'nearest':
                    student_last_layer_feat = resize(
                        input=student_last_layer_feat,
                        size=teacher_last_layer_feat.shape[2:],
                        mode='nearest',
                        align_corners=None,
                        warning=False
                    )
                else:
                    raise NotImplementedError

                depth_gt_resized = resize(
                    input=depth_gt,
                    size=teacher_last_layer_feat.shape[2:],
                    mode='nearest',
                    align_corners=None,
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
                student_losses.update(**distill_loss)
                
            student_losses['depth_info_teacher'] = teacher_losses['loss_depth']

        else:
            student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses = self.extract_feat(imgs, img_metas, depth_gt, **kwargs)
            

        if self.ema:
            ema_losses = self.extract_feat_ema(imgs, img_metas, depth_gt, **kwargs)
            student_losses['ema_info_depth'] = ema_losses['loss_depth'].detach()
        
        return student_losses
            

    # used in test, which is defined in depther model
    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass
        # return self.student_depther.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""

        if self.ema:
            mean_student = img.new_tensor(self.img_norm_cfg_student.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            std_student = img.new_tensor(self.img_norm_cfg_student.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            img_normed_student = (img - mean_student) / std_student
            return self.ema_model.simple_test(img_normed_student, img_meta, **kwargs)
        if self.val_model == 'student':
            mean_student = img.new_tensor(self.img_norm_cfg_student.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            std_student = img.new_tensor(self.img_norm_cfg_student.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            img_normed_student = (img - mean_student) / std_student
            return self.student_depther.simple_test(img_normed_student, img_meta, **kwargs)
        elif self.val_model == 'teacher':
            mean_teacher = img.new_tensor(self.img_norm_cfg_teacher.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            std_teacher = img.new_tensor(self.img_norm_cfg_teacher.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
            img_normed_teacher = (img - mean_teacher) / std_teacher
            return self.teacher_depther.simple_test(img_normed_teacher, img_meta, **kwargs)

        # return self.teacher_depther.simple_test(img, img_meta, **kwargs)