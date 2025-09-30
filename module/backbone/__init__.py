from module.backbone import resnet
from module.backbone.efficientnet import EfficientNetB0

BACKBONE = {
    'resnet18': resnet.ResNet,
    'resnet34': resnet.ResNet,
    'resnet50': resnet.ResNet,
    'resnet101': resnet.ResNet,
    'efficientnet_b0': EfficientNetB0,
}
