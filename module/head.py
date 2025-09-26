import torch.nn as nn
import torch.nn.functional as F
from module.init_weights import weights_init_normal


class ResRegLessCNN(nn.Module):
    def __init__(self, filter_num=32, scale=16):
        super(ResRegLessCNN, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(filter_num * scale, filter_num * scale // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(filter_num * scale // 2, 1)  # Single output for regression
        )
        self.apply(weights_init_normal)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x
