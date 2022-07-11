import numpy as np
from depth.datasets.builder import PIPELINES
import random
import mmcv

@PIPELINES.register_module()
class RandomCropV2(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping [(h, w)], when pick_model = False.
            Candidate size [(h1, w1), (h2, w2), ...., so on], when pick_model = False.
        pick_mode (bool): If apply select_mode
    """
    def __init__(self, crop_size, pick_mode=False):
        self.crop_size = crop_size
        self.pick_mode = pick_mode
        
        if self.pick_mode is not True:
            self.crop_size_h_min = crop_size[0][0]
            self.crop_size_h_max = crop_size[1][0]
            self.crop_size_w_min = crop_size[0][1]
            self.crop_size_w_max = crop_size[1][1]


    def random_select(self):
        if self.pick_mode:
            if len(self.crop_size) == 1:
                select_index = 0
            else:
                select_index = random.randint(0, len(self.crop_size) - 1)
            return self.crop_size[select_index]
        
        else:
            select_h = random.randint(self.crop_size_h_min, self.crop_size_h_max + 1)
            select_w = random.randint(self.crop_size_w_min, self.crop_size_w_max + 1)
            return (select_h, select_w)
        
    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        crop_size = self.random_select()
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop depth
        for key in results['depth_fields']:
            results[key] = self.crop(results[key], crop_bbox)

        results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class ResizeImg(object):
    """Resize images.

        # Apply to RandomCropV2, select_mode = True
        # (480, 640) -> (128, 160) ratio: 3.75, 4; factor: 32x4, 32x5 (max value)
        # (360, 512) -> (96, 128) ratio: 3.75, 4; factor: 32x3, 32x4
        # (240, 384) -> (64, 96) ratio: 3.75, 4; factor: 32x2, 32x3
        # (120, 256) -> (32, 64) ratio: 3.75, 4; factor: 32x1, 32x2

    """
    def __init__(self,
                 img_scale_ori=(480, 640),
                 img_scale_target=(128, 160)):

        self.scale_factor_h = img_scale_target[0] / img_scale_ori[0]
        self.scale_factor_w = img_scale_target[1] / img_scale_ori[1]

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(results['img'],
                                              (int(results['img'].shape[1] * self.scale_factor_w), 
                                              int(results['img'].shape[0] * self.scale_factor_h)),
                                              return_scale=True)
        
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, depth estimation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(scale_factor_h={self.scale_factor_h}, '
                     f'scale_factor_w={self.scale_factor_w})')
        return repr_str