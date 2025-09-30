import numpy as np

np.random.seed(0)
from models.reg_models.resnet import ResNetFusionTextNetRegression
from models.reg_models.FusionAttentionNet import ResNetFusionAttentionNetRegression

MODELS = {
    'ResNetFusionTextNetRegression': ResNetFusionTextNetRegression,
    'ResNetFusionAttentionNetRegression': ResNetFusionAttentionNetRegression,
}
