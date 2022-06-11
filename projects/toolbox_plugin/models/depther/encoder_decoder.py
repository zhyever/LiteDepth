import torch
from depth.models.builder import DEPTHER
from depth.ops import resize
from .encoder_decoder_downsample import DepthEncoderDecoderMobile

@DEPTHER.register_module()
class DepthEncoderDecoderMobileTF(DepthEncoderDecoderMobile):
    r'''
    used convert pytorch model to the tflite
    '''

    def forward(self, input):
        
        out = self.extract_feat(input)
        out = self.decode_head.forward(out, None)
        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        out = resize(
            input=out,
            size=(480, 640),
            mode='bilinear',
            align_corners=self.align_corners)

        return out
