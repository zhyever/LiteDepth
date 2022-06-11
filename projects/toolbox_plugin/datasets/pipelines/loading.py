import mmcv
import copy
import numpy as np
import os.path as osp
from PIL import Image
from depth.datasets.builder import PIPELINES

@PIPELINES.register_module()
class DepthLoadAnnotationsV2(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow',
                 with_x_grad=False,
                 with_y_grad=False,
                 with_bins_cls=False,
                 bins_num=8):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

        self.with_x_grad = with_x_grad
        self.with_y_grad = with_y_grad
        self.with_bins_cls = with_bins_cls
        self.bins_num = bins_num

    def generate_x_grad(self, depth_gt):
        init_x_grad = np.matrix(np.ones(depth_gt.shape) * np.inf)

        # convert invalid depth -> inf
        depth_gt[depth_gt == 0] = np.inf


        invalid_mask = depth_gt == 0
        print(invalid_mask)

        exit(100)

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')

        # avoid potential bugs
        _depth_gt = copy.deepcopy(depth_gt)
        if self.with_x_grad:
            depth_x_grad = self.generate_x_grad(_depth_gt)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str