from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.utils import build_from_cfg

from depth.utils import get_root_logger
from .pcgrad_optimizer import PCGradOptimizer
from mmcv.runner.optimizer import DefaultOptimizerConstructor

@OPTIMIZER_BUILDERS.register_module()
class PCGradOptimizerConstructor(DefaultOptimizerConstructor):
    """
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.ModuleDict({
        >>> })
        >>> optimizer = dict(type='Adam', lr=max_lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)
        >>> optim_builder = PCGradOptimizerConstructor(optimizer_cfg)
        >>> optimizer = optim_builder(model)
        >>> print(optimizer)
    """

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            optimizer = build_from_cfg(optimizer_cfg, OPTIMIZERS)
            return PCGradOptimizer(optimizer)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params
        optimizer = build_from_cfg(optimizer_cfg, OPTIMIZERS)
        return PCGradOptimizer(optimizer)