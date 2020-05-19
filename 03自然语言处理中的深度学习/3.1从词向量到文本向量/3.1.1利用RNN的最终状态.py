# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/5/18 -*-

# 利用RNN的最终状态将若干词向量变为一个向量的方法
import torch
import torch.nn as nn


class BiRNN(nn.Module):
    def __init__(self, word_dim, hidden_size):  # word_dim为词向量长度，hidden_size为RNN隐状态
        super(BiRNN, self).__init__()
        self.gru = nn.GRU(word_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)  # 输入的第一维张量为batch大小

    def forward(self, x):
        # 输入的x为batch组文本，长度为seq_len，词向量长度为word_dim，维度为batch*seq_len*word_dim
        # 输出为文本向量，维度为batch*(2*hidden_size)
        batch = x.shape[0]
        # output为每个单词对应的最后一层RNN的隐状态，维度为batch*seq_len*(2*hidden_size)
        # last_hidden为最终的RNN状态，维度为2*batch*hidden_size
        output, last_hidden = self.gru(x)
        print(last_hidden.transpose(0, 1).contiguous())
        return last_hidden.transpose(0, 1).contiguous().view(batch, -1)  # transpose只能操作2D矩阵的转置


# 测试
batch = 10
seq_len = 20
word_dim = 50
hidden_size = 100

x = torch.randn(batch, seq_len, word_dim)
bi_rnn = BiRNN(word_dim=word_dim, hidden_size=hidden_size)
res = bi_rnn(x)
print(res.shape)
print(res)
"""
tensor([[[ 0.0832, -0.0598,  0.2890,  ...,  0.1998, -0.0869, -0.3190],
         [ 0.0768,  0.3429,  0.2640,  ...,  0.2716, -0.1913,  0.2949]],

        [[-0.0393, -0.2930, -0.4658,  ..., -0.1160, -0.2333,  0.3248],
         [ 0.0447,  0.0867,  0.2720,  ...,  0.2934, -0.0334, -0.0835]],

        [[ 0.3302,  0.0884,  0.5714,  ..., -0.0582,  0.0012,  0.2477],
         [ 0.1471,  0.1168, -0.0972,  ..., -0.0783, -0.3259, -0.1885]],

        ...,

        [[ 0.1782, -0.2602, -0.0124,  ..., -0.1619, -0.0398,  0.0413],
         [-0.0626, -0.1427,  0.2677,  ...,  0.3754, -0.0121,  0.2151]],

        [[-0.2681, -0.3455,  0.2114,  ...,  0.2903, -0.2452,  0.3808],
         [ 0.0026, -0.0122,  0.2774,  ...,  0.1456,  0.1384,  0.0149]],

        [[-0.1343, -0.0865, -0.0146,  ..., -0.1296,  0.0488,  0.1067],
         [-0.0411,  0.2124,  0.1906,  ..., -0.2569, -0.3545,  0.1369]]],
       grad_fn=<CloneBackward>)
torch.Size([10, 200])
tensor([[ 0.0832, -0.0598,  0.2890,  ...,  0.2716, -0.1913,  0.2949],
        [-0.0393, -0.2930, -0.4658,  ...,  0.2934, -0.0334, -0.0835],
        [ 0.3302,  0.0884,  0.5714,  ..., -0.0783, -0.3259, -0.1885],
        ...,
        [ 0.1782, -0.2602, -0.0124,  ...,  0.3754, -0.0121,  0.2151],
        [-0.2681, -0.3455,  0.2114,  ...,  0.1456,  0.1384,  0.0149],
        [-0.1343, -0.0865, -0.0146,  ..., -0.2569, -0.3545,  0.1369]],
       grad_fn=<ViewBackward>)
"""
