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
import torch.nn.functional as F
import torch

from .modeling import BertEncoder, AttentionPooling, BertLayerNorm, PositionEmbeddings, \
    BertAttentionDag, BertIntermediateDag
from torch import nn
from transformers import logging
logging.set_verbosity_warning()
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class DAGAttention2D(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention2D, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors, mask=None):
        # concatenate the leaves and ancestors
        mask = mask.unsqueeze(2)
        x = torch.cat((leaves * mask, ancestors * mask), dim=-1)

        # Linear layer
        x = self.linear1(x)

        # relu activation
        x = torch.relu(x)

        # linear layer
        x = self.linear2(x)

        mask_attn = (1.0 - mask) * VERY_NEGATIVE_NUMBER
        x = x + mask_attn

        # softmax activation
        x = torch.softmax(x, dim=1)

        # weighted sum on ancestors
        x = (x * ancestors * mask).sum(dim=1)
        return x

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
    def __init__(self, config, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.config = config
        self.know_hidden_size = config.hidden_size
        self.embed_dag = None
        self.dag_attention = DAGAttention2D(2 * config.hidden_size, config.hidden_size)
        # self.encoder = BertEncoder(config)
        self.attention_know = BertAttentionDag(config)
        self.intermediate = BertIntermediateDag(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_know_coembedding = nn.Linear(200,48)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooling = AttentionPooling(config)
        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)
        self.embed_inputs = nn.Embedding(config.code_size, self.know_hidden_size)
        # self.encoder = KnowledgeEncoder(config)
        self.encoder_patient = BertEncoder(config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 64) #32
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.crossattention = CrossAttention(512, 256, 512) #350,256,350
        self.lstm = SentimentModel(400,512).to(device)
        self.dense_event = nn.Linear(512, 512)
        self.dense_visitpre = nn.Linear(config.intermediate_size, 512)
        self.b_gate = nn.Parameter(torch.zeros(512))
        self.sigmoid = nn.Sigmoid()

    def forward(self, code_x, divided, neighbors, lens, events, input_ids, code_mask, diagnosis_list):
        # for knowledge graph embedding 对leaves和non-leaf节点构建embddding maxtrix
        leaves_emb = self.embed_init(self.config.leaves_list)
        ancestors_emb = self.embed_init(self.config.ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.config.masks_list)
        padding = torch.zeros([1, self.know_hidden_size], dtype=torch.float32).to(self.config.device)
        dict_matrix = torch.cat([padding, dag_emb], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=False)
        # inputs embedding
        input_tensor = self.embed_inputs(input_ids)  # bs, visit_len, code_len, embedding_dim
        input_shape = input_tensor.shape
        inputs = input_tensor.view(-1, input_shape[2], input_shape[3])  # bs * visit_len, code_len, embedding_dim
        # entity embedding
        input_tensor_dag = self.embed_dag(input_ids)
        # bs * visit_len, code_len, embedding_dim
        inputs_dag = input_tensor_dag.view(-1, input_tensor_dag.shape[2], input_tensor_dag.shape[3])
        inputs_mask = code_mask.view(-1, input_tensor_dag.shape[2])  # bs * visit_len, code_len
        # attention mask for encoder
        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)  # bs * visit_len,1,1 code_len
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        hidden_states_output, hidden_states_dag_output, _, _ = \
            self.attention_know(inputs, extended_attention_mask, inputs_dag, None, None, output_attentions=True)
        intermediate_output = self.intermediate(hidden_states_output, hidden_states_dag_output)
        hidden_states = self.dense(intermediate_output)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_output + hidden_states_dag_output)
        # knowledge encoder
        visit_outputs, all_attentions = hidden_states, None  # 需要visit_outputs visit_outputs: bs * visit_len, max_visit_len, embedding_dim
        know_dignosis_code = visit_outputs.view(code_x.shape[0], -1, input_ids.shape[2], visit_outputs.shape[2])


        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        #model = BertModel.from_pretrained('/home/ubuntu/Zhang/Chet_events_LSTM_cuda/Chet_events/Bio_ClinicalBERT')
        '''
        第一层循环控制batch；第二层循环控制admission_num
        '''
        for code_x_i, divided_i, neighbor_i, len_i, events_i, know_dignosis_code_i, diagnosis_list_i \
                in zip(code_x, divided, neighbors, lens, events, know_dignosis_code, diagnosis_list):

            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, events_it, know_dignosis_code_it, diagnosis_list_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), events_i, know_dignosis_code_i, diagnosis_list_i)):
                # 使用nonzero函数获取所有非零元素的索引
                # nonzero_indexes = torch.nonzero(diagnosis_list_it)
                # # 从张量中获取非零元素
                # diagnosis_list_it = diagnosis_list_it[nonzero_indexes][:,0]
                # for index,item in enumerate(diagnosis_list_it):
                #     c_embeddings.data[item]=self.dense_know_coembedding(know_dignosis_code_it[index])
                    #c_embeddings.data[item]=know_dignosis_code_it[index]
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)#得到使用GNN处理后的有关诊断节点和邻居节点的上下文
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                #visit_attention_event = [] #attention between visit respresentation and event respresentation
                seqs = []#if tensor.eq(0).all():
                if events_it.eq(0).all():
                    combined_vector = output_it
                #output_it = output_it.expand(event_output.size(0), output_it.size(0))
                # 将文本向量和code向量沿着第0个维度拼接起来
                #combined_vector = torch.cat([event_output, output_it], dim=0)
                else:
                    for event in events_it:
                        seqs.append(len(event))
                    seqs = torch.Tensor(seqs).long()
                    # pdb.set_trace()
                    events_it = events_it.long()
                    event_output = self.lstm(events_it, seqs)  # (event_num,512)
                    event_output_sum = torch.sum(event_output, dim=0, keepdim=True)  # 沿着第一个维度相加，将(n,512)变成(1,512)
                    #output_it = torch.stack([output_it.unsqueeze(0), event_output_sum])
                    combined_vector = torch.cat((output_it.unsqueeze(0), event_output_sum), dim=0)
                    #output_it = output_it.squeeze(1)

                '''
                if event_output.size()[0] > 1:
                    # pdb.set_trace()
                    if output_it.size()[0] == 512:#
                        output_it = output_it.unsqueeze(0)
                    for event_output_i in event_output:
                        output_it = torch.vstack([output_it, event_output_i.unsqueeze(0)])
                    output_it = output_it.squeeze(1)
                else:
                    output_it = torch.stack([output_it.unsqueeze(0), event_output])
                    output_it = output_it.squeeze(1)
            #     visit_attention_event.append(output_it)
            # output_it = self.attention(torch.vstack(visit_attention_event))
            '''
                output_i.append(combined_vector)
            output_i = self.attention(torch.vstack(output_i))
            output.append(output_i)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output


'''
                if events_it != []:
                    for event in events_it:
                        #event = event[0]
                        seqs = []
                        seqs.append(len(event))
                        torch_event = pad_sequence([torch.from_numpy(np.array(x)) for x in event], batch_first=True).float()
                        torch_event = torch_event.long().to(device)
                        event_output = self.lstm(torch_event, seqs) #(event_num,512)
                        if event_output.size()[0]>1:
                            #pdb.set_trace()
                            if output_it.size()[0] == 512:
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
                else:
                    output_i.append(output_it)

                    ''''''
                    event_lens= len(event)
                    if event_lens > 512:
                        input_ids = torch.tensor(event[:512])
                        mask_len = [1]*event_lens
                        attention_mask = torch.tensor(mask_len[:512])
                    else:
                        input_ids = torch.tensor(event)
                        mask_len = [1] * event_lens
                        attention_mask = torch.tensor(mask_len)
                    pretrain_model_output = model(input_ids = input_ids.unsqueeze(0), attention_mask = attention_mask.unsqueeze(0))[1]
                    ''''''
                    #crossattention_output = self.crossattention(output_it.unsqueeze(0), event_output)
                    #visit_attention_event.append(crossattention_output)

            output_i = self.attention(torch.vstack(output_i))
            output.append(output_i)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output
'''