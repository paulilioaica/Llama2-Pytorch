import torch
import torch.nn as nn
import torch.nn.functional as F
from swiglu import SwiGLU
from grouped_query_attention import GroupedQueryAttention
from rms_norm import RMSNorm

    
class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden
        self.swiglu = SwiGLU(num_ffn_hidden)
        self.W_1 = nn.Linear(num_hidden, num_ffn_hidden)
        self.W_2 = nn.Linear(num_ffn_hidden, num_hidden)

    def forward(self, x):
        x = self.W_1(x)
        x = self.swiglu(x)
        x = self.W_2(x)
        return x



class LlamaModel(nn.Module):
    def __init__(self, num_layers, n_heads, num_kv_heads, seq_len, num_hidden) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([LlamaLayer(num_hidden, n_heads, num_kv_heads, seq_len) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.decoders:
            x = layer(x)
        return x


class LlamaLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, num_kv_heads, seq_len ) -> None:
        super().__init__()
        self.grouped_query_attention = GroupedQueryAttention(num_hidden=num_hidden, num_heads=num_heads, num_kv_heads=num_kv_heads, seq_len=seq_len, d_k=1)
        
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden=2*num_hidden)
        self.rms_norm1 = RMSNorm(num_hidden)
        self.rms_norm2 = RMSNorm(num_hidden)
    
    def forward(self, output_with_pos):
        x = self.rms_norm1(output_with_pos)

        # attention
        x = self.grouped_query_attention(x, x, x)

        #add and norm
        x_after_attention = x + output_with_pos
        
        x = self.rms_norm2(x)

        #add and norm
        x = self.feed_forward(x)

        # residual connection
        x = x + x_after_attention
        return x

class Llama2(nn.Module):
    def __init__(self, decoder_layers_num, num_hidden, num_heads, num_kv_heads, seq_len, vocab_size) -> None:
        super().__init__()
        self.model = LlamaModel(decoder_layers_num, num_heads, num_kv_heads, seq_len, num_hidden)
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.linear = nn.Linear(num_hidden, vocab_size)
        self.rms_norm = RMSNorm(num_hidden)

    def forward(self, x):
        #embeddings
        x = self.embedding(x)

        #forward pass
        output = self.model(x)
        # rms norm
        output = self.rms_norm(output)

        output = self.linear(output)

        return output
