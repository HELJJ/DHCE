import math

import torch
from torch import nn
from math import sqrt

class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size): #value_size 350
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output

class CrossAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v): #256 256 256  #512, 256, 512
        super(CrossAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(512, dim_k, bias=False)
        self.linear_v = nn.Linear(512, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x ,y):
        # x: batch, n, dim_in
        #batch, n, dim_in = x.shape
        #assert dim_in == self.dim_in
        #(output_it, event_output) (1,512) (512,)
        #其中的问题(query)是 output_it  文本(value)是event_output q和k不同 k与value相同
        x = x.unsqueeze(0) #(512,) -> (1,512)
        y = y.unsqueeze(0)
        q = self.linear_q(x)  # batch, n, dim_k (1,1,512)
        k = self.linear_k(y)  # batch, n, dim_k (1,3,512)
        v = self.linear_v(y)  # batch, n, dim_v ()

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n (1,1,512) (1,512,3)->(1,1,3)
        dist = torch.softmax(dist, dim=-1)  # batch, n, n (1,3,1) (1,3,512) (1,)

        att = torch.matmul(dist, v) #(1,1,3) (1,3,512)->(1,1,512)
        return att
