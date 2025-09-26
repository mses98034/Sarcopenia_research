from module.backbone import resnet

BACKBONE = {
    'resnet18': resnet.ResNet,
    'resnet34': resnet.ResNet,
    'resnet50': resnet.ResNet,
    'resnet101': resnet.ResNet,
}
