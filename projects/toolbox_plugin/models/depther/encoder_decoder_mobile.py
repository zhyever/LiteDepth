import torch
import torch.nn as nn

from depth.models.depther import DepthEncoderDecoder
from depth.models.builder import DEPTHER
from depth.ops import resize

from collections import OrderedDict
import torch.distributed as dist


import mmcv
import numpy as np
from depth.utils import colorize


@DEPTHER.register_module()
class DepthEncoderDecoderMobile(DepthEncoderDecoder):
    r'''
    used in mobileAI challenge
    '''

    def __init__(self,
                 pcgrad=False,
                 gt_target_shape=(480, 640),
                 pixel_shuffle=False,
                 **kwarg):
        super(DepthEncoderDecoderMobile, self).__init__(**kwarg)
        self.gt_target_shape = gt_target_shape
        self.pixel_shuffle = pixel_shuffle
        if self.pixel_shuffle:
            self.pixel_shuffle_layer = nn.PixelUnshuffle(4)
        self.pcgrad = pcgrad

    def init_weights(self):
        super(DepthEncoderDecoderMobile, self).init_weights()

        if self.pixel_shuffle:
            first_conv = self.backbone.timm_model.conv_stem
            first_conv.in_channels = first_conv.in_channels * 16
            first_conv.weight = nn.Parameter(torch.cat([first_conv.weight] * 16, 1) / 16)

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
                size=self.gt_target_shape,
                mode='nearest')
        return out

    def extract_feat(self, img):
        """Extract features from images."""

        if self.pixel_shuffle:
            img = self.pixel_shuffle_layer(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
    
        return x


    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)

        # split losses and images
        real_losses = {}
        log_imgs = {}
        for k, v in losses.items():
            if 'img' in k:
                log_imgs[k] = v
            else:
                real_losses[k] = v

        loss, log_vars = self._parse_losses(real_losses, self.pcgrad)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            log_imgs=log_imgs)

        return outputs

    @staticmethod
    def _parse_losses(losses, pcgrad=False):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        if pcgrad:
            loss = [_value for _key, _value in log_vars.items()
                   if 'loss' in _key]
            loss_sum = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)
            log_vars['loss'] = loss_sum

        else:
            loss = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)
            log_vars['loss'] = loss
            
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    format_only=False):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The depth estimation results.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        depth = result[0]

        if show:
            mmcv.imshow(img, win_name, wait_time)

        if format_only:
            if out_file is not None:
                np.save(out_file, depth) # only save the value.
        else:
            if out_file is not None:
                # depth = colorize(depth, vmin=self.student_depther.decode_head.min_depth, vmax=self.student_depther.decode_head.max_depth)
                depth = colorize(depth, vmin=None, vmax=None)
                mmcv.imwrite(depth.squeeze(), out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result depth will be returned')
            return depth