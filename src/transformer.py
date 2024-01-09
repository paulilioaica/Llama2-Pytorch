from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from swiglu import SwiGLU
from grouped_query_attention import GroupedQueryAttention
from rms_norm import RMSNorm
# Feed forward definition
    
class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden
        self.swiglu = SwiGLU()
        self.W_1 = nn.Linear(num_hidden, num_ffn_hidden)
        self.W_2 = nn.Linear(num_ffn_hidden, num_hidden)

    def forward(self, x):
        return self.W_2(self.swiglu(self.W_1(x)))


# Transformer definition

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, n_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([TransformerDecoderLayer(num_hidden, n_heads, seq_len) for i in range(num_layers)])

    def forward(self, x, encoder_output):
        for layer in self.decoders:
            x = layer(x, encoder_output)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, seq_len, num_kv_heads) -> None:
        super().__init__()
        self.multihead_attention_masked = GroupedQueryAttention(num_hidden=num_hidden, num_heads=num_heads, num_kv_heads=num_kv_heads, seq_len=seq_len, d_k=1)
        self.multihead_attention = GroupedQueryAttention(num_hidden=num_hidden, num_heads=num_heads, num_kv_heads=num_kv_heads, seq_len=seq_len, d_k=1)
        
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden=2*num_hidden)
        self.rms_norm1 = RMSNorm(num_hidden)
        self.rms_norm2 = RMSNorm(num_hidden)
    
    def forward(self, output_with_pos, encoder_output):
        x = self.rms_norm1(output_with_pos)
        # masked attention
        x = self.multihead_attention_masked(x, x, x)
        #add and norm
        x_after_attention = x + output_with_pos
        
        x = self.rms_norm2(x)

        #add and norm
        x = self.feed_forward(x)

        # residual connection
        x = x + x_after_attention
        return x

class Llama2(nn.Module):
    def __init__(self, decoder_layers_num, num_hidden, num_heads, seq_len, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.decoder = TransformerDecoder(decoder_layers_num, num_heads, seq_len, num_hidden)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.rms_norm = RMSNorm(embedding_dim)

    def forward(self, x):
        #embeddings
        x = self.embedding(x)

        #forward pass
        output = self.decoder(x)
        # rms norm
        output = self.rms_norm(output)

        output = self.linear(output)
        output = F.softmax(output, dim=-1)

        return output
