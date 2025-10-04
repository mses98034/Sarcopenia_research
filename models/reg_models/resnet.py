import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel
from module.backbone import BACKBONE
from module.backbone.resnet import get_backbone_feature_dim, RESNET_CONFIGS
from models.reg_models.TextNet import TextNetFeature
from torch.nn.functional import normalize
import torch
from module.non_local import NLBlockND
from module.head import ResRegLessCNN
import torchvision.models as models
from module.torchcam.methods import SmoothGradCAMpp


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention


class ResNetFusionTextNetRegression(BaseModel):
    def __init__(self, backbone, n_channels, use_pretrained=True, pretrained=None, config=None):
        # Handle backward compatibility with old 'pretrained' parameter
        if pretrained is not None:
            use_pretrained = pretrained

        super(ResNetFusionTextNetRegression, self).__init__(backbone, n_channels, pretrained=use_pretrained)

        # Use modern parameter name
        self.backbone = BACKBONE[backbone](backbone=backbone, use_pretrained=use_pretrained)
        self.text_net = TextNetFeature(backbone=backbone, n_channels=n_channels, pretrained=use_pretrained)

        # Dynamic feature dimension based on backbone architecture
        self.feature_dim = get_backbone_feature_dim(backbone)
        self.filter_num = 32
        self.text_dim = self.text_net.hidden[-1]

        self.atten = Self_Attn(self.feature_dim + self.text_dim)

        # Robust, dynamic scale determination based on feature dimension
        if self.feature_dim == 512: # For ResNet18, ResNet34
            self.scale1 = 16
        elif self.feature_dim == 2048: # For ResNet50, ResNet101, etc.
            self.scale1 = 64
        elif self.feature_dim == 1280: # For EfficientNet-B0
            self.scale1 = 40 # 1280 / 32 (filter_num) = 40
        else:
            # A generic fallback for other potential backbones
            # This might need adjustment if a new backbone's feature_dim is not divisible by filter_num
            self.scale1 = self.feature_dim // self.filter_num

        # Image-only regression head using ResRegLessCNN
        self.image_regressor = ResRegLessCNN(filter_num=self.filter_num, scale=self.scale1)

        self.nlb4 = NLBlockND(in_channels=self.filter_num * self.scale1, mode='concatenate', dimension=2,
                              bn_layer=True)

        self.aw = nn.Parameter(torch.zeros(2))
        self.ca = nn.Parameter(torch.zeros(1))

        self.softmax2d = nn.Softmax2d()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fusion_projector = nn.Sequential(
            nn.Linear(self.feature_dim + self.text_dim, self.feature_dim + self.text_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim + self.text_dim, self.feature_dim // 2),
            nn.Dropout(0.2),
        )

        # Multi-modal fusion regression head using ResRegLessCNN
        # Calculate the input dimension for fusion features
        fusion_dim = self.feature_dim + self.text_dim  # 512 + 64 = 576
        # We need to adapt ResRegLessCNN for this dimension
        # Since ResRegLessCNN expects filter_num * scale, we calculate equivalent scale
        fusion_scale = fusion_dim // self.filter_num  # 576 // 32 = 18
        self.fusion_regressor = ResRegLessCNN(filter_num=self.filter_num, scale=fusion_scale)

        # --- CAM Generator Initialization ---
        # Load an independent, pretrained ResNet18 model as CAM generator
        # Use new weights parameter if available
        try:
            from torchvision.models import ResNet18_Weights
            if use_pretrained:
                self.cam_generator = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.cam_generator = models.resnet18(weights=None)
        except ImportError:
            # Fallback for older torchvision versions
            self.cam_generator = models.resnet18(pretrained=use_pretrained)

        # CAM Generator training control
        self.train_cam_generator = getattr(config, 'train_cam_generator', False) if config else False

        # Always use eval mode for stable BatchNorm (using ImageNet global statistics)
        self.cam_generator.eval()

        if not self.train_cam_generator:
            # Freeze mode: requires_grad=False (weights won't be updated)
            for param in self.cam_generator.parameters():
                param.requires_grad = False
        # Otherwise: requires_grad=True (weights can be updated, but BatchNorm still uses global stats)

        # Initialize SmoothGradCAMpp extractor, bound to generator's last conv block ('layer4')
        self.cam_extractor = SmoothGradCAMpp(self.cam_generator, 'layer4')
        # --- End of CAM Generator Initialization ---

        # CAM enhancement control parameter
        self.cam_enhancement = getattr(config, 'cam_enhancement', True) if config else True

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, x, text=None, text_included=False, cam=None):
        # --- CAM Enhancement Steps ---
        if self.cam_enhancement:
            # Ensure CAM generator is on the same device as input tensor (CPU/GPU)
            self.cam_generator.to(x.device)

            try:
                # 1. Use CAM generator to generate coarse attention map (Xm)
                # Note: CAM generator may be frozen or trainable (controlled by train_cam_generator config)
                # Gradient flow is allowed regardless for visualization (plot.py GradCAM)
                scores = self.cam_generator(x)

                # 2. Extract CAM. For regression task, model has single output, so class_idx fixed to 0
                # if text_included and text is not None:
                #     activation_map = self.cam_extractor(class_idx=0, scores=scores, text=text)
                # else:
                #     activation_map = self.cam_extractor(class_idx=0, scores=scores)
                activation_map = self.cam_extractor(class_idx=0, scores=scores)
                Xm = activation_map[0]  # cam_extractor returns list, take first element

                # 3. Use bilinear interpolation to resize CAM to match input image size (H, W)
                Xm_resized = F.interpolate(Xm.unsqueeze(1), size=(x.size(2), x.size(3)),
                                           mode='bilinear', align_corners=False)

                # 4. Apply paper's enhancement formula: Xf = X * (1 + sigmoid(Xm))
                # Add channel dimension to Xm_resized for element-wise multiplication
                Xf = x * (1 + torch.sigmoid(Xm_resized))
            except Exception as e:
                if self.training:
                    print(f"Warning: CAM extraction failed during training: {e}, using original input")
                # Fallback to original input when CAM fails
                Xf = x
        else:
            # No CAM enhancement, use original input
            Xf = x
        # --- End of CAM Enhancement Steps ---

        # Pass enhanced image `Xf` to backbone network instead of original `x`
        h = self.backbone(Xf)
        nlb = self.nlb4(h[-1])

        if text_included and text is not None:
            text_w = self.aw[0]
            text_feature = self.text_net(text)  # Clinical features shape = [batch, 64, 1]
            text = self.text_net.num(text.permute(0, 2, 1))
            text_feature = text_feature + text_w * self.softmax2d(text_feature) * text
            text_feature = text_feature.unsqueeze(-1).expand(-1, 64, 7, 7)

            vis_feature = nlb
            x_fusion = torch.cat((vis_feature, text_feature), dim=1)  # (batch, 512+64, 7, 7)

            out, _ = self.atten(x_fusion)
            out_i = self.avgpool(out)
            x_fusion = torch.flatten(out_i, 1)
            contrs_learn = x_fusion
            self.feats = x_fusion

            out = self.fusion_regressor(x_fusion.unsqueeze(-1).unsqueeze(-1))  # Add spatial dimensions for ResRegLessCNN
        else:
            # Image-only mode
            out = self.image_regressor(h[-1])  # ResRegLessCNN handles pooling internally
            x_pool = F.adaptive_avg_pool2d(h[-1], (1, 1))
            x_pool = x_pool.view(x_pool.size(0), -1)
            contrs_learn = x_pool
            self.feats = x_pool
        
        z_i = normalize(self.fusion_projector(contrs_learn), dim=1)
        return out, z_i

    def get_feats(self):
        return self.feats

if __name__ == '__main__':
    # Test regression model
    rft_reg = ResNetFusionTextNetRegression('resnet18', 3, True)
    print(rft_reg)
