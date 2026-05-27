import torch
import torch.nn as nn
import torch.nn.functional as F


class ReshapeConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        return self.activation(x)


class NorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        return self.activation(x)


class NorConv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        return self.activation(x)


class TCRNetwork(nn.Module):
    """Temporal consistency residual used after Gaussian rasterization."""

    def __init__(self):
        super().__init__()
        self.tcr = ReshapeConv()
        self.tcr_edge = NorConv()
        self.tcr_refine = NorConv()

    def forward(self, x):
        img_tensor = x[0]
        grad_x = torch.gradient(img_tensor, dim=1)
        grad_y = torch.gradient(img_tensor, dim=0)

        grad_x_x = torch.gradient(grad_x[0], dim=1)
        grad_y_y = torch.gradient(grad_y[0], dim=0)

        second_order_edge = torch.abs(grad_x_x[0]) + torch.abs(grad_y_y[0])
        second_order_edge = torch.unsqueeze(second_order_edge, dim=0)
        residual_input = torch.cat((torch.unsqueeze(x[0], dim=0), second_order_edge), dim=0)

        residual = self.tcr(residual_input)
        residual = self.tcr_edge(residual)
        residual = self.tcr_refine(residual)
        return torch.cat([residual] * 3, dim=0)
