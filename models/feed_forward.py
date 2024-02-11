import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, device=device, lr=1e-4):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.device = device
        self.to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        outputs = self.model(state)
        
        return outputs
    
    def get_distribution(self, state):
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        return dist