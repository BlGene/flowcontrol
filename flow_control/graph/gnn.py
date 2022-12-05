import os
import torch
from torch import nn

from torchvision.models import resnet18, ResNet18_Weights

class DisjGNN(nn.Module):
    def __init__(self, params):
        super(DisjGNN, self).__init__()

        self.img_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.img_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_encoder.fc = nn.Linear(self.fov_encoder.fc.in_features, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x, edge_index, edge_time_delta) -> torch.Tensor:

        out_x = self.img_encoder.forward(x)

        # Construct edge features and concatenate
        x_i, x_j = out_x[edge_index[0,:]], out_x[edge_index[1,:]]

        edge_feats = torch.cat([x_i, x_j])

        out_edge_attr = self.mlp.forward(edge_feats)

        return out_edge_attr
