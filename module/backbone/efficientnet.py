import torch.nn as nn
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, backbone, use_pretrained=True):
        super(EfficientNetB0, self).__init__()
        # Load pretrained EfficientNet-B0
        if use_pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        original_model = models.efficientnet_b0(weights=weights)
        
        # We only need the feature extraction part of the model
        self.features = original_model.features

    def forward(self, x):
        # The original model returns a list of feature maps at different stages.
        # Here, we simplify and return only the final feature map from the feature extractor.
        # This is a 4D tensor, e.g., [batch_size, 1280, 7, 7]
        final_features = self.features(x)
        
        # To maintain compatibility with the existing ResNet implementation which returns a list,
        # we wrap our single output tensor in a list.
        return [final_features]
