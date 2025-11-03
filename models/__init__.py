import numpy as np

np.random.seed(0)
from models.reg_models.resnet import ResNetFusionTextNetRegression
from models.reg_models.FusionAttentionNet import ResNetFusionAttentionNetRegression
from models.reg_models.HybridAttentionNet import ResNetHybridAttentionNetRegression

MODELS = {
    'ResNetFusionTextNetRegression': ResNetFusionTextNetRegression,
    'ResNetFusionAttentionNetRegression': ResNetFusionAttentionNetRegression,
    'ResNetHybridAttentionNetRegression': ResNetHybridAttentionNetRegression,
}
