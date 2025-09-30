import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from collections import namedtuple

# Import ResNet weights for different architectures
try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
    WEIGHTS_AVAILABLE = True
except ImportError:
    # Fallback for older torchvision versions
    WEIGHTS_AVAILABLE = False

# Unified ResNet configuration supporting multiple architectures
RESNET_CONFIGS = {
    'resnet18': {
        'model': models.resnet18,
        'weights': ResNet18_Weights.IMAGENET1K_V1 if WEIGHTS_AVAILABLE else None,
        'feature_dim': 512
    },
    'resnet34': {
        'model': models.resnet34,
        'weights': ResNet34_Weights.IMAGENET1K_V1 if WEIGHTS_AVAILABLE else None,
        'feature_dim': 512
    },
    'resnet50': {
        'model': models.resnet50,
        'weights': ResNet50_Weights.IMAGENET1K_V1 if WEIGHTS_AVAILABLE else None,
        'feature_dim': 2048
    },
    'resnet101': {
        'model': models.resnet101,
        'weights': ResNet101_Weights.IMAGENET1K_V1 if WEIGHTS_AVAILABLE else None,
        'feature_dim': 2048
    }
}

def get_backbone_feature_dim(backbone):
    """Get the feature dimension for a given backbone"""
    if backbone in RESNET_CONFIGS:
        return RESNET_CONFIGS[backbone]['feature_dim']
    elif backbone == 'efficientnet_b0':
        return 1280
    else:
        supported_backbones = list(RESNET_CONFIGS.keys()) + ['efficientnet_b0']
        raise ValueError(f"Unsupported backbone: {backbone}. Supported: {supported_backbones}")

class ResNet(nn.Module):
    def __init__(self, backbone, use_pretrained=True, pretrained=None):
        super(ResNet, self).__init__()

        # Handle backward compatibility with old 'pretrained' parameter
        if pretrained is not None:
            use_pretrained = pretrained

        if backbone not in RESNET_CONFIGS:
            raise ValueError(f"Unsupported backbone: {backbone}. Supported: {list(RESNET_CONFIGS.keys())}")

        config = RESNET_CONFIGS[backbone]

        # Use new weights parameter if available, fall back to pretrained for compatibility
        if WEIGHTS_AVAILABLE and use_pretrained:
            resnet = config['model'](weights=config['weights'])
        elif use_pretrained:
            resnet = config['model'](pretrained=True)
        else:
            resnet = config['model'](weights=None) if WEIGHTS_AVAILABLE else config['model'](pretrained=False)
        self.topconvs = nn.Sequential(
            OrderedDict(list(resnet.named_children())[0:3]))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.topconvs(x)
        layer0 = x
        x = self.max_pool(x)
        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x

        res_outputs = namedtuple("SideOutputs", ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'])
        out = res_outputs(layer0=layer0, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        return out

if __name__ == '__main__':
    RN = ResNet('resnet18', False)
    print(RN)