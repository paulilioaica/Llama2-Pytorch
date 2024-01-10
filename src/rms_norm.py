import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-15):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        
        x_normed = x / (norm_x + self.eps)

        return x_normed * self.weight