
import torch
import torch.nn as nn

class PearsonLoss(nn.Module):
    """ 
    Calculates the Pearson correlation coefficient loss, which is 1 - r.
    Aims to maximize the Pearson correlation (r) by minimizing the loss.
    """
    def __init__(self, eps=1e-6):
        super(PearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure inputs are flattened
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate means
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)

        # Calculate Pearson correlation coefficient
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.eps)
        
        # Loss is 1 - r
        return 1 - cost

class CCCLoss(nn.Module):
    """
    Calculates the Concordance Correlation Coefficient (CCC) loss.
    CCC measures both correlation and deviation from the y=x line.
    It's a more robust measure of agreement than Pearson.
    """
    def __init__(self, eps=1e-6):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure inputs are flattened
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate means
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)

        # Calculate variances
        var_pred = torch.var(y_pred)
        var_true = torch.var(y_true)

        # Calculate covariance
        covariance = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

        # Calculate CCC
        ccc = (2 * covariance) / (var_pred + var_true + (mean_pred - mean_true)**2 + self.eps)

        # Loss is 1 - CCC
        return 1 - ccc
