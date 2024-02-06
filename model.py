import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, obs_space, out_dim, hidden_dim=64):
        super().__init__()
        self.in_dim = obs_space[0]
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, batch_size=32):
        result = torch.empty(
            size=(x.shape[0], self.out_dim), dtype=torch.float, device=torch.device("cpu"))
        for start_idx in range(0, x.shape[0], batch_size):
            batch = x[start_idx: start_idx + batch_size].to(self.fc1.weight.device)
            batch = self.fc1(batch)
            batch = F.relu(batch)
            batch = self.fc2(batch)
            batch = F.relu(batch)
            batch = self.fc3(batch)

            result[start_idx: start_idx + batch_size] = batch.to(torch.device("cpu"))
        return result


class ConvolutionalNN(nn.Module):
    def __init__(self, obs_space, out_dim):
        super().__init__()
        h, w, c = obs_space
        self.out_dim = out_dim

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

    def forward(self, x, batch_size=32):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        result = torch.empty(size=(x.shape[0], self.out_dim), device=torch.device("cpu"))
        
        for start_idx in range(0, x.shape[0], batch_size):
            batch = x[start_idx: start_idx + batch_size].to(self.conv1.weight.device)
            batch = batch.to(torch.float) / 255
            batch = batch.permute(0, 3, 1, 2)

            batch = self.ln1(self.pool1(self.relu1(self.conv1(batch))))
            batch = self.ln2(self.pool2(self.relu2(self.conv2(batch))))
            batch = self.ln3(self.pool3(self.relu3(self.conv3(batch))))

            batch = batch.view(batch.shape[0], -1)

            batch = self.fc2(self.relu4(self.fc1(batch)))
            
            result[start_idx: start_idx + batch_size] = batch.to(torch.device("cpu"))

        return result
