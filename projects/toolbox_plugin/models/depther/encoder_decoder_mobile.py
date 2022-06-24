import torch
from depth.models.depther import DepthEncoderDecoder
from depth.models.builder import DEPTHER
from depth.ops import resize

@DEPTHER.register_module()
class DepthEncoderDecoderMobile(DepthEncoderDecoder):
    r'''
    used in mobileAI challenge
    '''

    def __init__(self,
                 downsample_ratio=4,
                 **kwarg):
        super(DepthEncoderDecoderMobile, self).__init__(**kwarg)
        self.downsample_ratio = downsample_ratio

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
