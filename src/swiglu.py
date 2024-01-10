import torch.nn as nn
import torch 

class SiLU(nn.Module): 
    def __init__(self):
        super().__init__()
    def forward(self, x): 
        return x * torch.sigmoid(x) 

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.silu = SiLU()
        self.expand_linear = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, x):
        x = self.expand_linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.silu(gate)  