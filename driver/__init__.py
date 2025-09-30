import torch
import torchvision.transforms as tvt
from sarcopenia_data.auto_augment import AutoAugment

from torch.optim import SGD, Adam, AdamW

OPTIM = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_local = tvt.Compose([
    tvt.ToPILImage(),
    AutoAugment(),
    tvt.ToTensor(),
])

transform_test = tvt.Compose([
    tvt.ToPILImage(),
    tvt.ToTensor(),
])
