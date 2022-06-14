import torch
from depth.models.depther import DepthEncoderDecoder
from depth.models.builder import DEPTHER
from depth.ops import resize

@DEPTHER.register_module()
class DepthEncoderDecoderMobile(DepthEncoderDecoder):
    r'''
    used in mobileAI challenge
    '''


    def extract_feat(self, img):
        """Extract features from images."""

        # x4 downsample the input image for speed up
        img = resize(input=img, size=(img.shape[-2] // 4, img.shape[-1] // 4), mode='bilinear', align_corners=self.align_corners)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
    
        return x
