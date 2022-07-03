from .l1_loss import SmoothL1Loss, L1Loss
from .mse_loss import MSELoss
from .cwd_loss import ChannelWiseDivergence
from .kd_loss import KnowledgeDistillationKLDivLoss
from .similarity_mse_loss import SimilarityMSELoss
from .custom_distill import CustomDistll
from .memory_loss import StudentSegContrast
from .grad_loss import GradDepthLoss
from .ssim_loss import SSIMDepthLoss

__all__ = [
    'SmoothL1Loss', 'L1Loss',
    'MSELoss', 'ChannelWiseDivergence',
    'KnowledgeDistillationKLDivLoss',
    'SimilarityMSELoss',
    'CustomDistll',
    'StudentSegContrast',
    'GradDepthLoss',
    'SSIMDepthLoss'
]