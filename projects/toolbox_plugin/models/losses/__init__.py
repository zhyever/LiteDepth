from .l1_loss import SmoothL1Loss, L1Loss
from .distill_margin_loss import MarginLoss
from .cwd_loss import ChannelWiseDivergence
from .kd_loss import KnowledgeDistillationKLDivLoss
from .similarity_mse_loss import SimilarityMSELoss
from .custom_distill import CustomDistll
from .memory_loss import StudentSegContrast
from .grad_loss import GradDepthLoss
from .ssim_loss import SSIMDepthLoss
from .vnl_loss import VNLLoss
from .pair_wise_loss import PairMSELoss
from .robust_loss import RobustLoss
from .auto_weight_loss import AutoReweightLoss
from .si_rmse_loss import SiRMSELoss
from .grad_error_loss import GradDepthErrorLoss

__all__ = [
    'SmoothL1Loss', 'L1Loss',
    'MarginLoss', 'ChannelWiseDivergence',
    'KnowledgeDistillationKLDivLoss',
    'SimilarityMSELoss',
    'CustomDistll',
    'StudentSegContrast',
    'GradDepthLoss',
    'SSIMDepthLoss',
    'VNLLoss',
    'PairMSELoss',
    'RobustLoss',
    'AutoReweightLoss',
    'SiRMSELoss',
    'GradDepthErrorLoss'
]