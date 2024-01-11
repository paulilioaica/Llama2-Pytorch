import torch
import torch.nn as nn
# Positional encoding definition

class RotaryEncodings(nn.Module):
    def __init__(self, seq_len, num_hidden, dropout=0.1, base=10_000,):
        super().__init__()      
        self.base = base
        self.num_hidden = num_hidden
        self.sequence_length = seq_len

        theta = 1. / (self.base ** (torch.arange(0, self.num_hidden).float() / self.num_hidden))  
        index = torch.arange(seq_len).float()

        phase = index.unsqueeze(-1) * theta
        
        self.sin = phase.sin()[:, None, None, :]
        self.cos = phase.cos()[:, None, None, :]

    def neg(self, x):
        return torch.cat([-x[:, :, :, self.num_hidden//2:], x[:, :, :, :self.num_hidden//2]], dim=-1)
    
    def forward(self, x):
        neg_x = self.neg(x)
        x = x * self.cos[:x.shape[0]].to(x.device) + neg_x * self.sin[:neg_x.shape[0]].to(x.device)
        return x
