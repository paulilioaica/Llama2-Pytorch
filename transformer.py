from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Vanilla Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_k = d_k

        self.W_q = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def get_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, key, values, dropout=0.1, mask=None):

        # Passing through linear layer
        # Reshaping expanded to n_heads
        query = self.W_q(query).view(-1, self.num_heads, self.seq_len, self.num_hidden)
        key = self.W_k(key).view(-1, self.num_heads, self.seq_len, self.num_hidden)
        values = self.W_v(values).view(-1, self.num_heads, self.seq_len, self.num_hidden)

        # Q * K_T
        QK_T = torch.matmul(query,  key.mT)

        # QK_T / sqrt(dk)
        QK_T = QK_T / math.sqrt(self.d_k)

        if mask:
            mask = self.get_mask(self.seq_len)
            QK_T = QK_T.masked_fill(mask == 1, float('-inf'))

        # softmax(QK_T / sqrt(d_k)
        attention_scores = self.softmax(QK_T)
        
        #dropout
        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, values)  
        # Reshape and apply output linear layer  
        output = output.transpose(1, 2).contiguous().view(-1, self.seq_len, self.num_heads * self.num_hidden)  
        output = self.W_o(output)  
          
        return output  


class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden

        self.W_1 = nn.Linear(num_hidden, num_ffn_hidden)
        self.W_2 = nn.Linear(num_ffn_hidden, num_hidden)

    def forward(self, x):
        return self.W_2(F.relu(self.W_1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len) -> None:
        super().__init__()
        self.multihead_attention = MultiHeadAttention(num_hidden=num_hidden, num_heads=num_heads, seq_len=seq_len, d_k=1)
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden=2*num_hidden)
        self.layer_norm1 = nn.LayerNorm(num_hidden)
        self.layer_norm2 = nn.LayerNorm(num_hidden)
    
    def forward(self, input_with_pos):
        #attention add and norm
        x = self.multihead_attention(input_with_pos, input_with_pos, input_with_pos)
        x += input_with_pos
        x_add_norm = self.layer_norm1(x)

        # add norm feedforward
        x_final = self.feed_forward(x_add_norm)
        x_final += x_add_norm
        x_final = self.layer_norm2(x_final)
        return x_final

class TransformerDecoder(nn.Module):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, n_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList([TransformerEncoderLayer(num_hidden, n_heads, seq_len) for i in range(num_layers)])
    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)





class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
