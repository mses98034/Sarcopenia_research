import numpy as np

np.random.seed(0)
from models.reg_models.resnet import ResNetFusionTextNetRegression

MODELS = {
    'ResNetFusionTextNetRegression': ResNetFusionTextNetRegression,
}
