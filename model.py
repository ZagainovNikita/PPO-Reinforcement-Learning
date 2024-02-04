import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, obs_space, out_dim, hidden_dim=64):
        super().__init__()
        in_dim = obs_space[0]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ConvolutionalNN(nn.Module):
    def __init__(self, obs_space, out_dim):
        super().__init__()
        h, w, c = obs_space

        self.conv1 = nn.Conv2d(c, 16, 3, 1, 1)
        c = 16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        h //= 2
        w //= 2
        self.ln1 = nn.LayerNorm((c, h, w))
        
        self.conv2 = nn.Conv2d(c, 32, 3, 1, 1)
        c = 32
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        h //= 2
        w //= 2
        self.ln2 = nn.LayerNorm((c, h, w))
        
        self.conv3 = nn.Conv2d(c, 64, 3, 1, 1)
        c = 64
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        h //= 2
        w //= 2
        self.ln3 = nn.LayerNorm((c, h, w))
        
        self.fc1 = nn.Linear(c * h * w, 1024)
        self.relu4 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, out_dim)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        x = self.ln1(self.pool1(self.relu1(self.conv1(x))))
        x = self.ln2(self.pool2(self.relu2(self.conv2(x))))
        x = self.ln3(self.pool3(self.relu3(self.conv3(x))))
        
        x = x.view(x.shape[0], -1)
        
        x = self.fc2(self.relu4(self.fc1(x)))
        
        return x