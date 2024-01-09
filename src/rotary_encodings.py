import torch
import torch.nn as nn
# Positional encoding definition

class RotaryEncodings(nn.Module):
    def __init__(self, seq_len, device, num_hidden, dropout=0.1, base=10_000,):
        super().__init__()      
        self.base = base
        self.num_hidden = num_hidden

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)  
        index = torch.arange(seq_len).float().to(device)
        phase = theta * index
        self.sin = phase.sin()[:, None, None, :]
        self.cos = phase.cos()[:, None, None, :]

    def neg(self, x):
        num_hidden_half = self.num_hidden // 2
        return torch.cat(-x[:, :, :, num_hidden_half:], x[:, :, :, :num_hidden_half])
    
    def forward(self, x):
        neg_x = self.neg(x)
        x = x * self.cos[:x.shape[0]] + neg_x * self.sin[:neg_x.shape[0]]
        return x
