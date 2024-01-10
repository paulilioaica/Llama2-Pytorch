import torch.nn as nn
import torch.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate) 