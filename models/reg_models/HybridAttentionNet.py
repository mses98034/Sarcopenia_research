import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reg_models.resnet import ResNetFusionTextNetRegression, Self_Attn
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

class ResNetHybridAttentionNetRegression(ResNetFusionTextNetRegression):
    """
    Hybrid Attention Network that combines both Gated Attention and Self-Attention mechanisms.

    Architecture Flow:
    1. Image features (512D) + Text features (64D)
    2. Gated Attention Fusion (concatenate + gating mechanism)
    3. Self-Attention (spatial attention on fused features)
    4. AvgPool + Flatten → Regression
    """
    def __init__(self, backbone, use_pretrained, n_channels, config=None):
        # Initialize the parent class, which builds the backbone, text_net, etc.
        super().__init__(backbone, use_pretrained, n_channels, config)

        # The input channel dimension for the fusion module is the sum of
        # image feature map channels and text feature map channels.
        fusion_in_channels = self.feature_dim + self.text_dim # e.g., 512 + 64 = 576

        # Replace the parent's attention with our hybrid attention modules
        self.gated_fusion = GatedAttentionFusion(in_channels=fusion_in_channels)
        self.self_attention = Self_Attn(in_dim=fusion_in_channels)

        # The parent class already created self.fusion_regressor, we can reuse it.

    def forward(self, x, text=None, text_included=False, cam=None):
        # CAM Enhancement (inherited from parent)
        if self.cam_enhancement:
            self.cam_generator.to(x.device)
            try:
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
            # Process text features (inherited from parent)
            text_w = self.aw[0]
            text_feature = self.text_net(text)  # Clinical features shape = [batch, 64, 1]
            text = self.text_net.num(text.permute(0, 2, 1))
            text_feature = text_feature + text_w * self.softmax2d(text_feature) * text
            text_feature_map = text_feature.unsqueeze(-1).expand(-1, self.text_dim, vis_feature.size(2), vis_feature.size(3))

            # Step 1: Gated Attention Fusion
            gated_fused_map = self.gated_fusion(vis_feature, text_feature_map) # Shape: [batch, 576, 7, 7]

            # Step 2: Self-Attention on gated fused features
            self_attn_out, _ = self.self_attention(gated_fused_map) # Shape: [batch, 576, 7, 7]

            # Pool, flatten, and predict
            out_i = self.avgpool(self_attn_out)
            x_fusion_flat = torch.flatten(out_i, 1)
            contrs_learn = x_fusion_flat
            self.feats = x_fusion_flat

            # Use the parent's fusion regressor
            out = self.fusion_regressor(x_fusion_flat.unsqueeze(-1).unsqueeze(-1))

        else:
            # Image-only mode (inherited from parent)
            out = self.image_regressor(h[-1])
            x_pool = F.adaptive_avg_pool2d(h[-1], (1, 1))
            x_pool = x_pool.view(x_pool.size(0), -1)
            contrs_learn = x_pool
            self.feats = x_pool

        # Project for contrastive loss (inherited from parent)
        z_i = F.normalize(self.fusion_projector(contrs_learn), dim=1)
        return out, z_i
