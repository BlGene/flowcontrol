import os
import torch
from torch import nn


from torchvision.models import resnet18, ResNet18_Weights

class DisjGNN(nn.Module):
    def __init__(self):
        super(DisjGNN, self).__init__()

        self.img_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.img_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, 64)

        self.sim_mlp = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )


        self.pos_mlp = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.rot_mlp = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x, edge_index):

        x = x.view(-1, 4, 256, 256)
        out_x = self.img_encoder.forward(x)

        # Construct edge features and concatenate
        x_i, x_j = out_x[edge_index[0,:]], out_x[edge_index[1,:]]

        edge_feats = torch.cat([x_i, x_j], dim=1)

        out_edge_attr = self.sim_mlp.forward(edge_feats)
        out_pos_diff = self.pos_mlp.forward(edge_feats)
        out_rot_diff = self.rot_mlp.forward(edge_feats)

        return out_edge_attr, out_x, out_pos_diff, out_rot_diff
