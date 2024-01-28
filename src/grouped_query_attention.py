from typing import Any
from rotary_encodings import RotaryEncodings
import torch.nn as nn
import torch
import math

class KVCacheMemory():
    def __init__(self, num_kv_heads, seq_len, num_hidden):
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.num_hidden = num_hidden
        self.curr_pos = 0
        self.init_cache()
        # self.device = device

    def reset_pos(self):
        self.curr_pos = 0
        self.init_cache()

    def init_cache(self, batch_size = None):
        # self.batch_size = batch_size
        self.k_cached = torch.zeros((1,  self.num_kv_heads, self.seq_len, self.num_hidden))
        self.q_cached = torch.zeros((1, self.num_kv_heads, self.seq_len, self.num_hidden))

    def update(self, k, q):
        self.k_cached[:, :, self.curr_pos : self.curr_pos + 1, :] = k
        self.q_cached[:, :, self.curr_pos : self.curr_pos + 1, :] = q
        self.curr_pos += 1

    def __call__(self):
        return self.k_cached[:, :, :self.curr_pos, :], self.q_cached[:, :,  :self.curr_pos, :]

        

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, num_kv_heads, seq_len, d_k, dropout=0.1) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_rep = self.num_heads // self.num_kv_heads

        self.rotary_encodings = RotaryEncodings(seq_len, num_hidden)


        self.seq_len = seq_len
        self.d_k = d_k
        
        #caching KV for inference time
        self.cache = KVCacheMemory(num_kv_heads, seq_len, num_hidden)

        self.W_q = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, num_kv_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_kv_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mask = self.get_mask(self.seq_len)
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.tril(torch.ones(size, size, device=device), diagonal=-1)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, keys, values, mask=False):
        # Reshaping expanded to n_heads or n_kv_heads
        seq_len, num_hidden = query.shape[1], query.shape[2]
        
        query = self.W_q(query).view(-1, self.num_heads, seq_len, num_hidden)
        keys = self.W_k(keys).view(-1, self.num_kv_heads, seq_len, num_hidden)
        values = self.W_v(values).view(-1, self.num_kv_heads, seq_len, num_hidden)

        # shape [batch, seq_len, kv_heads, hidden]
        rope_keys = self.rotary_encodings(keys) 
        rope_values = self.rotary_encodings(values)


        # if evaluation, [batch, seq_len = 1, ....], so we need to cache the KV    
        if not self.training:
            # in this case keys and values span the whole sequence but we cache only the last one
            # so we have the keys and values for last token in the sequence 
            self.cache.update(rope_keys, rope_values)

            # then we need to get the whole cached keys and values        
            rope_keys, rope_values = self.cache()


        # bring them to the same shape as original key and values
        rope_keys = rope_keys.repeat((1, self.num_rep, 1, 1))
        rope_values = rope_values.repeat((1, self.num_rep, 1, 1))

        # Q * K_T, in this case its [batch, 1, heads, seq_len] * [batch, seq_len, heads, hidden]
        QK_T = torch.matmul(query,  rope_keys.mT)

        # QK_T / sqrt(dk)
        QK_T = QK_T / math.sqrt(self.d_k)

        # mask
        if mask and self.training:
            QK_T = QK_T.masked_fill(mask == 0, float("-inf"))

        # softmax(QK_T / sqrt(d_k)
        attention_scores = self.softmax(QK_T)

        #dropout
        if self.training:
            attention_scores = self.dropout(attention_scores)

        output = torch.matmul(attention_scores, rope_values)  

        # Reshape and apply output linear layer  
        output = output.transpose(1, 2).contiguous().view(-1, seq_len, self.num_heads * num_hidden)  
        output = self.W_o(output)  
          
        return output  
