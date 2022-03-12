import math

import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim, class_num):
        super(Net, self).__init__()
        self.class_num = class_num
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(dim, 500, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(500, 500, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(500, 2000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(2000, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, dim, bias=True),
            nn.Sigmoid(),
        )
        self.cluster_layer = nn.Linear(10, class_num, bias=False)
        self.cluster_center = torch.rand([class_num, 10], requires_grad=False).cuda()

    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def cluster(self, z):
        return self.cluster_layer(z)

    def init_cluster_layer(self, alpha, cluster_center):
        self.cluster_layer.weight.data = 2 * alpha * cluster_center

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (
            F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha
        )

    def predict(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = torch.argmin(distance, dim=1)
        return prediction

    def set_cluster_centroid(self, mu, cluster_id, alpha):
        self.cluster_layer.weight.data[cluster_id] = 2 * alpha * mu


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class NetConv(Net):
    def __init__(self, channel, inner_dim, class_num):
        super(NetConv, self).__init__(dim=inner_dim, class_num=class_num)
        self.class_num = class_num
        self.inner_dim = inner_dim
        self.kernel_size = int(math.sqrt(inner_dim / 16))
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(inner_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, inner_dim, bias=True),
            Reshape(16, self.kernel_size, self.kernel_size),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
