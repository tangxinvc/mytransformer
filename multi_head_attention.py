import math
from typing import Optional

import torch
from torch import nn as nn

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x) 
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
        # i , j represent the sequence length
        # b represents batch_size
        # h, d represent heads and d_k individually.
    
    def forward(self, *, query: torch.tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape
        # query, key, value: seq_len, batch_size, d_model
        
        if mask is not None:
        # mask has shape [seq_len_q, seq_len_k, batch_size]
            assert mask.shape[0] == 1 or mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]
            assert mask.shape[2] == 1 or mask.shape[2] == query.shape[1]
        mask = mask.unsqueeze(-1)  # [seq_len_q, seq_len_k, batch_size, 1]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        scores = self.get_scores(query, key)
        scales *= self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum('ijbh,jbh->ibh', attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)
        
             