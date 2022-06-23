from depth.models.builder import DEPTHER
from depth.models.depther import BaseDepther
from depth.models import build_depther
import torch
import mmcls.models
import torch.nn as nn
from depth.ops import resize
from depth.models.builder import build_loss

@DEPTHER.register_module()
class DistillWrapper(BaseDepther):
    def __init__(self,
                 teacher_depther_cfg,
                 student_depther_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 teacher_select_de_index=(0, ),
                 student_select_de_index=(0, ),
                 distill_loss=None,
                 pretrained=None,
                 super_resolution=False,
                 align_corners=True,
                 layer_weights=(1, ),
                 img_norm_cfg_teacher=None,
                 img_norm_cfg_student=None,
                 val_model='student'):
        super(DistillWrapper, self).__init__(init_cfg)

        self.teacher_depther_cfg = teacher_depther_cfg
        self.student_depther_cfg = student_depther_cfg

        self.teacher_depther = build_depther(teacher_depther_cfg)
        self.student_depther = build_depther(student_depther_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # freeze the teacher model
        self.freeze(self.teacher_depther)

        # teacher_encoder_channels = teacher_depther_cfg.decode_head.in_channels[::-1]
        # teacher_decoder_channels = teacher_depther_cfg.decode_head.up_sample_channels[::-1]
        # student_encoder_channels = student_depther_cfg.decode_head.in_channels[::-1]
        # student_decoder_channels = student_depther_cfg.decode_head.up_sample_channels[::-1]

        self.teacher_select_de_index = teacher_select_de_index
        self.student_select_de_index = student_select_de_index
        self.layer_weights = layer_weights

        # self.feat_proj = nn.Conv2d(
        #     student_depther_cfg.decode_head.up_sample_channels[student_select_de_index],
        #     teacher_depther_cfg.decode_head.up_sample_channels[teacher_select_de_index], 1)
        
        self.distill_loss_cfg = distill_loss
        if isinstance(distill_loss, list):
            self.distill_loss = [build_loss(i) for i in distill_loss]
        else:
            self.distill_loss = build_loss(distill_loss)

        self.align_corners = align_corners

        self.super_resolution = super_resolution

        self.img_norm_cfg_teacher = img_norm_cfg_teacher
        self.img_norm_cfg_student = img_norm_cfg_student

        self.val_model = val_model

    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(DistillWrapper, self).init_weights()

    def extract_feat(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for extract features from images."""

        mean_teacher = imgs.new_tensor(self.img_norm_cfg_teacher.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std_teacher = imgs.new_tensor(self.img_norm_cfg_teacher.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        img_normed_teacher = (imgs - mean_teacher) / std_teacher
        mean_student = imgs.new_tensor(self.img_norm_cfg_student.mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std_student = imgs.new_tensor(self.img_norm_cfg_student.std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        img_normed_student = (imgs - mean_student) / std_student

        imgs_4x_teacher = resize(input=img_normed_teacher, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=self.align_corners)
        imgs_4x_student = resize(input=img_normed_student, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=self.align_corners)

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

        return teacher_decoder_feats_act, teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
            student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses

    def forward_train(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for Forward function for training."""

        teacher_decoder_feats_act, teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
            student_decoder_feats_act, student_encoder_feats, student_decoder_feats, student_losses = self.extract_feat(imgs, img_metas, depth_gt, **kwargs)

        for idx, (idx_t, idx_s, w) in enumerate(zip(self.teacher_select_de_index, self.student_select_de_index, self.layer_weights)):
            teacher_last_layer_feat = teacher_decoder_feats[-(idx_t + 1)]
            student_last_layer_feat = student_decoder_feats[-(idx_s + 1)]

            student_last_layer_feat = resize(
                input=student_last_layer_feat,
                size=teacher_last_layer_feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False
            )
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
                    distill_loss['distill_loss_{}_{}'.format(self.distill_loss_cfg[i].type, idx_t)] = loss_temp * self.train_cfg.distill_loss_weight[i]
            else:
                distill_loss = self.distill_loss(student_last_layer_feat, teacher_last_layer_feat, depth_gt_resized)
                distill_loss = {'distill_loss_{}'.format(idx_t): distill_loss * w}
            student_losses.update(**distill_loss)
            
        student_losses['depth_loss_teacher'] = teacher_losses['loss_depth']

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