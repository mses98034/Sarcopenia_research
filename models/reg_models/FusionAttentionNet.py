import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reg_models.resnet import ResNetFusionTextNetRegression
from models.reg_models.TextNet import TextNetFeature 
from module.backbone.resnet import get_backbone_feature_dim

class GatedAttentionFusion(nn.Module):
    """
    A Gated Attention mechanism for fusing 4D feature maps (image + text).
    It learns a "gate" to scale the combined features.
    """
    def __init__(self, in_channels):
        super(GatedAttentionFusion, self).__init__()
        self.in_channels = in_channels
        # A simple 1x1 convolution to generate the gate
        self.gating_network = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image_feature_map, text_feature_map):
        # Concatenate along the channel dimension
        fused_map = torch.cat((image_feature_map, text_feature_map), dim=1)
        # Generate the gate
        gate = self.gating_network(fused_map)
        # Apply the gate
        attended_map = fused_map * gate
        return attended_map

class ResNetFusionAttentionNetRegression(ResNetFusionTextNetRegression):
    """
    This model uses a Gated Attention mechanism to fuse features.
    It correctly operates on 4D feature maps before pooling and regression.
    """
    def __init__(self, backbone, use_pretrained, n_channels, config=None):
        # Initialize the parent class, which builds the backbone, text_net, etc.
        super().__init__(backbone, use_pretrained, n_channels, config)

        # The input channel dimension for the fusion module is the sum of
        # image feature map channels and text feature map channels.
        fusion_in_channels = self.feature_dim + self.text_dim # e.g., 512 + 64

        # Replace the parent's attention and regressor with our new ones
        self.fusion_module = GatedAttentionFusion(in_channels=fusion_in_channels)
        
        # The final regressor remains the same as the parent's fusion_regressor
        # The parent class already created self.fusion_regressor, we can just reuse it.
        # No need to redefine self.fc_out or self.fusion_regressor if the parent's is suitable.

    def forward(self, x, text=None, text_included=False, cam=None):
        # Use the parent's forward pass to get the initial feature maps
        # This is a simplified version of the parent's forward logic up to the fusion point.
        
        # CAM Enhancement (copied from parent)
        if self.cam_enhancement:
            self.cam_generator.to(x.device)
            try:
                # CAM generator may be frozen or trainable (controlled by train_cam_generator config)
                # Gradient flow is allowed regardless for visualization (plot.py GradCAM)
                scores = self.cam_generator(x)
                activation_map = self.cam_extractor(class_idx=0, scores=scores)
                Xm = activation_map[0]
                Xm_resized = F.interpolate(Xm.unsqueeze(1), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                Xf = x * (1 + torch.sigmoid(Xm_resized))
            except Exception:
                Xf = x
        else:
            Xf = x

        # Get image feature maps from backbone
        h = self.backbone(Xf)
        vis_feature = self.nlb4(h[-1]) # Shape: [batch, 512, 7, 7]

        if text_included and text is not None:
            # Get text feature map (expanded to match image spatial dims)
            text_feature_map = self.text_net(text)
            text_feature_map = text_feature_map.unsqueeze(-1).expand(-1, self.text_dim, vis_feature.size(2), vis_feature.size(3))

            # Fuse using our new Gated Attention module
            fused_map = self.fusion_module(vis_feature, text_feature_map) # Shape: [batch, 576, 7, 7]
            
            # Pool, flatten, and predict (similar to parent)
            out_i = self.avgpool(fused_map)
            x_fusion_flat = torch.flatten(out_i, 1)
            contrs_learn = x_fusion_flat
            self.feats = x_fusion_flat
            
            # Use the parent's fusion regressor
            out = self.fusion_regressor(x_fusion_flat.unsqueeze(-1).unsqueeze(-1))

        else:
            # Image-only mode (copied from parent)
            out = self.image_regressor(h[-1])
            x_pool = F.adaptive_avg_pool2d(h[-1], (1, 1))
            x_pool = x_pool.view(x_pool.size(0), -1)
            contrs_learn = x_pool
            self.feats = x_pool
        
        # Project for contrastive loss (copied from parent)
        z_i = F.normalize(self.fusion_projector(contrs_learn), dim=1)
        return out, z_i
