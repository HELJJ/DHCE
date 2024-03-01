import numpy as np
import torch
from torch import nn
import pdb
import torch
from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention, CrossAttention
from models.LSTM import SentimentModel
from transformers import BertTokenizer,BertModel
from collections import OrderedDict
from transformers import  AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn
from transformers import logging
logging.set_verbosity_warning()

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        #self.graph_layer1 = GraphLayer(adj, code_size, graph_size)
        #self.graph_layer = GraphLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.attention = DotProductAttention(hidden_size, 64) #32
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.crossattention = CrossAttention(512, 256, 512) #350,256,350
        self.lstm = SentimentModel(400,270).to(device)

    def forward(self, code_x, divided, neighbors, lens, events):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        #model = BertModel.from_pretrained('/home/ubuntu/Zhang/Chet_events_LSTM_cuda/Chet_events/Bio_ClinicalBERT')
        '''
        第一层循环控制batch；第二层循环控制admission_num
        '''
        for code_x_i, divided_i, neighbor_i, len_i, events_i in zip(code_x, divided, neighbors, lens, events):
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, events_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), events_i)):
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)#得到使用GNN处理后的有关诊断节点和邻居节点的上下文
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                visit_attention_event = [] #attention between visit respresentation and event respresentation
                seqs = []
                for event in events_it:
                    seqs.append(len(event))
                seqs = torch.Tensor(seqs).long()
                events_it = events_it.long()
                event_output = self.lstm(events_it, seqs)  # (event_num,512) (3,270)
                #new_event_output = event_output.unsqueeze(1).view(-1,1,270)
                if event_output.size()[0] > 1:
                    # pdb.set_trace()
                    if output_it.size()[0] == 270:#
                        output_it = output_it.unsqueeze(0)
                    for event_output_i in event_output:
                        output_it = torch.vstack([output_it, event_output_i.unsqueeze(0)])
                    output_it = output_it.squeeze(1)
                else:
                    output_it = torch.stack([output_it.unsqueeze(0), event_output])
                    output_it = output_it.squeeze(1)
                visit_attention_event.append(output_it)

            output_it = self.attention(torch.vstack(visit_attention_event))
            output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i))
            output.append(output_i)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output


