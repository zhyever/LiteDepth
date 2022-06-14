from depth.models.builder import DEPTHER
from depth.models.depther import BaseDepther
from depth.models import build_depther
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
                 teacher_select_de_index=0,
                 student_select_de_index=0,
                 distill_loss=None,
                 pretrained=None,
                 align_corners=True):
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

        self.feat_proj = nn.Conv2d(
            teacher_depther_cfg.decode_head.up_sample_channels[teacher_select_de_index], 
            student_depther_cfg.decode_head.up_sample_channels[student_select_de_index], 1)
        
        self.distill_loss = build_loss(distill_loss)
        self.align_corners = align_corners

    
    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(DistillWrapper, self).init_weights()

    def extract_feat(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for extract features from images."""
        imgs = resize(input=imgs, size=(imgs.shape[-2] // 4, imgs.shape[-1] // 4), mode='bilinear', align_corners=self.align_corners)
        teacher_encoder_feats = self.teacher_depther.backbone(imgs)
        student_encoder_feats = self.student_depther.backbone(imgs)

        teacher_decoder_feats, teacher_outputs, teacher_losses = \
            self.teacher_depther.decode_head.forward_train(imgs, 
                                                           teacher_encoder_feats, 
                                                           img_metas, 
                                                           depth_gt, 
                                                           self.train_cfg, 
                                                           return_immediately=True, 
                                                           **kwargs)
        student_decoder_feats, student_outputs, student_losses = \
            self.student_depther.decode_head.forward_train(imgs, 
                                                           student_encoder_feats, 
                                                           img_metas, 
                                                           depth_gt, 
                                                           self.train_cfg, 
                                                           return_immediately=True, 
                                                           **kwargs)

        return teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
            student_encoder_feats, student_decoder_feats, student_losses

    def forward_train(self, imgs, img_metas, depth_gt, **kwargs):
        """Placeholder for Forward function for training."""

        teacher_encoder_feats, teacher_decoder_feats, teacher_losses, \
            student_encoder_feats, student_decoder_feats, student_losses = self.extract_feat(imgs, img_metas, depth_gt, **kwargs)

        # print("teacher encoder:")
        # for i in teacher_encoder_feats:
        #     print(i.shape)
        
        # print("student encoder:")
        # for i in student_encoder_feats:
        #     print(i.shape)
        
        # print("teacher decoder:")
        # for i in teacher_decoder_feats:
        #     print(i.shape)

        # print("student decoder:")
        # for i in student_decoder_feats:
        #     print(i.shape)

        teacher_last_layer_feat = teacher_decoder_feats[-(self.teacher_select_de_index + 1)]
        student_last_layer_feat = student_decoder_feats[-(self.student_select_de_index + 1)]
        teacher_last_layer_feat_proj = self.feat_proj(teacher_last_layer_feat)
        teacher_last_layer_feat_proj_resized = resize(
            input=teacher_last_layer_feat_proj,
            size=student_last_layer_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False
        )
        distill_loss = self.distill_loss(teacher_last_layer_feat_proj_resized, student_last_layer_feat)
        distill_loss = {'distill_loss': distill_loss}
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
        return self.student_depther.aug_test(imgs, img_metas, **kwargs)
        # return self.student_depther.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        return self.student_depther.simple_test(img, img_meta, **kwargs)
        # return self.teacher_depther.simple_test(img, img_meta, **kwargs)