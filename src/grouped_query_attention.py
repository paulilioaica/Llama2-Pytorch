from typing import Any
from rotary_encodings import RotaryEncodings
import torch.nn as nn
import torch
import math

class KVCacheMemory():
    def __init__(self, batch_size, num_heads, seq_len, num_hidden, device):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.num_hidden = num_hidden

    def init_cache(self, batch_size):
        self.batch_size = batch_size
        self.k_cached = torch.zeros((batch_size, self.seq_len, self.num_heads, self.num_hidden )).to(self.device)
        self.q_cached = torch.zeros((batch_size, self.seq_len, self.num_heads, self.num_hidden)).to(self.device)

    def update(self, k, q, curr_pos):
        self.k_cached[:self.batch_size, curr_pos : curr_pos + k.shape[1]] = k
        self.q_cached[:self.batch_size, curr_pos : curr_pos + q.shape[1]] = q
    
    def __call__(self, start_pos):
        return self.k_cached[:self.batch_size, :start_pos + self.seq_len], self.q_cached[:self.batch_size, :start_pos + self.seq_len]  

        

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, num_kv_heads, seq_len, d_k) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_rep = self.num_heads // self.num_kv_heads

        self.rotary_encodings = RotaryEncodings(seq_len, self.device, num_hidden)


        self.seq_len = seq_len
        self.d_k = d_k
        
        #caching KV for inference time
        self.cache = KVCacheMemory(num_heads, seq_len, num_hidden)

        self.W_q = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, num_kv_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_kv_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.mask = self.get_mask(self.seq_len)
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, key, values, dropout=0.1, mask=None):
        # Reshaping expanded to n_heads or n_kv_heads
        query = self.W_q(query).view(-1, self.num_heads, self.seq_len, self.num_hidden)
        key = self.W_k(key).view(-1, self.num_kv_heads, self.seq_len, self.num_hidden)
        values = self.W_v(values).view(-1, self.num_kv_heads, self.seq_len, self.num_hidden)


        # shape [batch, seq_len, kv_heads, hidden]
        rope_query = self.rotary_encodings(query) 
        rope_key = self.rotary_encodings(key)

        rope_query = rope_query.repeat((1, 1, self.num_kv_heads * self.num_rep, 1))
        rope_key = rope_key.repeat((1, 1, self.num_kv_heads * self.num_rep, 1))

        # Q * K_T
        QK_T = torch.matmul(query,  key.mT)

        # QK_T / sqrt(dk)
        QK_T = QK_T / math.sqrt(self.d_k)

        if mask:
            QK_T = QK_T.masked_fill(self.mask == 1, float('-inf'))

        # softmax(QK_T / sqrt(d_k)
        attention_scores = self.softmax(QK_T)
        
        #dropout
        if self.train():
            attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, values)  
        # Reshape and apply output linear layer  
        output = output.transpose(1, 2).contiguous().view(-1, self.seq_len, self.num_heads * self.num_hidden)  
        output = self.W_o(output)  
          
        return output  
