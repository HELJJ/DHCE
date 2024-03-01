import torch
import os
import random
import re #split使用
#import gensim # word2vec预训练加载
#import jieba #分词
from torch import nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
#from zhconv import convert #简繁转换
# 变长序列的处理
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
# from tqdm import tqdm

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.hidden_dim = hidden_dim
        #self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        #self.embeddings.weight.requires_grad = True

        self.embedding = nn.Embedding(28996, 400)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.5, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, batch_seq_len, hidden=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeds = self.embedding(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True, enforce_sorted=False)
        batch_size, seq_len = input.size()
        if hidden is None:
            c_0 = torch.zeros((2, batch_size, 512)).to(device) #MIMIC-III 256
            h_0 = torch.zeros((2, batch_size, 512)).to(device)  # [rnn层数,batch_size,hidden_size]
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        output, _ = pad_packed_sequence(output, batch_first=True)
        # output = self.dropout(torch.tanh(self.fc1(output)))
        output_last = output[:,0,-512:]
        # output = torch.tanh(self.fc2(output))
        # output = self.fc3(output)
        # last_outputs = self.get_last_output(output, batch_seq_len)
        #         output = output.reshape(batch_size * seq_len, -1)
        return output_last

    def get_last_output(self, output, batch_seq_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i] - 1]  # index 是长度 -1
        last_outputs = last_outputs.to(output.device)
        return last_outputs