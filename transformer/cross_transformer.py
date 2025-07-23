from torch import nn

import math
import torch

from .tools import *
        

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_q2 = nn.Linear(d_model, d_model)
        self.w_k2 = nn.Linear(d_model, d_model)
        self.w_v2 = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.w_concat2 = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q2, k2, v2 = self.w_q2(k), self.w_k2(q), self.w_v2(q)

        q, k, v = self.split(q), self.split(k), self.split(v)
        q2, k2, v2 = self.split(q2), self.split(k2), self.split(v2)
        batch_size, head, length, d_tensor = v2.size()

        out = self.attention(q, k, v, mask=mask)
        out2 = self.attention(q2,k2,v2,mask=mask)

        out,out2 = self.concat(out),self.concat(out2)
        out,out2 = self.w_concat(out),self.w_concat2(out2)

        return out,out2
    def split(self, tensor):

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model,n_head)
        self.ffn = PositionwiseFeedForward(d_model, ffn_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        N,C,T,V = x.size()
        x = x.permute(0, 3, 2,1).reshape(N * V, T, C) 
        N,C,T,V = y.size()
        y = y.permute(0, 3, 2,1).reshape(N * V, T, C)
        out1,out2 = self.attn(x, y, y, mask=mask)
        out1,out2 = self.dropout1(out1), self.dropout1(out2)
        out1,out2 = self.norm1(out1+x),self.norm1(out2+y)

        x,y = out1,out2
        out1,out2 = self.ffn(out1),self.ffn(out2)
        out1,out2 = self.dropout2(out1),self.dropout2(out2)
        out1,out2 = self.norm2(out1+x),self.norm2(out2+y)

        out1 = out1.view(N, V, T, -1).permute(0, 3, 2, 1)
        out2 = out2.view(N, V, T, -1).permute(0, 3, 2, 1)
        return out1,out2


