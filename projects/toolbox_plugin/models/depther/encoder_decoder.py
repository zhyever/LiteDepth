import torch
from turtle import forward
from depth.models.depther import DepthEncoderDecoder
from depth.models.builder import DEPTHER
from depth.ops import resize

@DEPTHER.register_module()
class DepthEncoderDecoderTF(DepthEncoderDecoder):
    def forward(self, input):
        
        x = self.extract_feat(input)

        out = self.decode_head.forward(x, None)

        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)

        out = resize(
            input=out,
            size=input.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return out