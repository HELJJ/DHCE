import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.u = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, text, code):
        # 转换text张量的维度
        text = text.unsqueeze(0)
        #text = text.permute(0, 2, 1)  # 将维度转换为(batch_size, L, D)
        text = self.W(text) + self.b
        text = torch.tanh(text)

        # 计算注意力分数
        code = code.unsqueeze(1).unsqueeze(0) # 将维度转换为(batch_size, 1, L)
        #score = torch.bmm(text, code.transpose(1, 2)) #(bs,D,L) (bs,L,1)
        score = torch.bmm(text, code) #(bs,D,L) (bs,L,1)
        score = torch.softmax(score, dim=1)
        # 加权融合
        text_att = torch.bmm(text.transpose(1, 2), score).squeeze(2) #(1,512,3) (1,3,1) (1.512.1)

        # 返回融合后的特征向量
        return text_att #(1,512)