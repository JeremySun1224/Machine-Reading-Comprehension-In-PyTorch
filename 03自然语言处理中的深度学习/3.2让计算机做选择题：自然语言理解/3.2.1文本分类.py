# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/5/20 -*-

"""
1、分类问题可以抽象成有K个种类，模型需要判定将每个输入文本分到哪个类别中。
因此，用于分类的深度学习网络也被称为判定模型。而自然语言中与分类相关的任
务被称为自然语言理解（NLU）


2、用于分类的神经网络一般由以下部分构成：
    首先，对文本进行分词，在获得词向量后，经过全连接、RNN、CNN等神经元层后，
    使用上一节介绍的方法得到一个文本向量d。这个网络被称为编码器（encoder）。

    接下来，利用这个文本向量的信息计算出K个数字，即模型认为这段文本属于每个
    类别的分数。

    计算得分通常使用一个d_dim*K的全连接层实现，其中d_dim为文本向量d的维度。
    这个全连接层也是整个模型的输出层。

    因此，模型有K个输出分数，利用交叉熵损失函数计算当前分数和标准答案之间的
    差异，并进行求导优化。

3、对于文本分类问题，如果系统需要模型“不属于任何一类”的情况，可以有两种解决办法：

    1.在验证数据上枚举得到最合适的阈值（threshold），当所有类别的得分概率都低
    于这个阈值时，输出“不属于任何一类”。

    2.可以将“不属于任何一类”作为第K+1类进行处理。
"""

import torch
import torch.nn as nn
import torch.optim as optim


class CNNMaxPooling(nn.Module):
    def __init__(self, word_dim, window_size, out_channels):  # output_dim表示CNN输出通道数
        super(CNNMaxPooling, self).__init__()
        # 1个输入通道，out_channels个输出通道，过滤器大小为window_size*word_dim
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size, word_dim))

    # 输入x为batch组文本，长度为seq_len，词向量长度为word_dim，维度为batch*seq_len*word_dim
    # 输出res为所有文本向量，每个向量的维度为out_channels
    def forward(self, x):
        x_unsqueeze = x.unsqueeze(1)  # 变成单通道，结果维度为batch*1*seq_len*word_dim
        x_cnn = self.cnn(x_unsqueeze)  # CNN，结果维度为batch*out_channels*new_seq_len*1
        x_cnn_result = x_cnn.squeeze(3)  # 删除最后一维，结果维度为batch*out_channels*new_seq_len
        res, _ = x_cnn_result.max(2)  # 最大池化，遍历最后一维求最大值，结果维度为batch*out_channels
        return res


class NLUNet(nn.Module):
    # word_dim是词向量维度，window_size为CNN窗口长度，out_channels为CNN输出通道数，K为类别个数
    def __init__(self, word_dim, window_size, out_channels, K):
        super(NLUNet, self).__init__()
        self.cnn_max_pool = CNNMaxPooling(word_dim=word_dim, window_size=window_size, out_channels=out_channels)  # CNN和最大池化
        self.linear = nn.Linear(in_features=out_channels, out_features=K)  # 输出为全连接层

    # x为输入tensor，维度为batch*seq_len*word_dim；输出class_score，维度为batch*K
    def forward(self, x):
        doc_embed = self.cnn_max_pool(x)  # 文本向量，结果维度是batch*out_channels
        class_score = self.linear(doc_embed)  # 分类分数，结果维度是batch*K
        return class_score


# 测试
K = 3
net = NLUNet(word_dim=10, window_size=3, out_channels=15, K=K)
x = torch.randn(30, 5, 10, requires_grad=True)  # 共30个序列，每个序列长度为5， 词向量长度为10
y = torch.LongTensor(30).random_(0, K)  # 共30个真实分类，类别为0~K-1的整数
optimizer = optim.SGD(net.parameters(), lr=0.01)
res = net(x)  # res大小为batch*K
# PyTorch自带交叉熵函数，包含计算softmax
loss_func = nn.CrossEntropyLoss()
loss = loss_func(res, y)
print('loss1 = {loss1}'.format(loss1=loss))
optimizer.zero_grad()
loss.backward()
optimizer.step()
res = net(x)
loss = loss_func(res, y)
print('loss2 = {loss2}'.format(loss2=loss))

"""
loss1 = 1.095941185951233
loss2 = 1.0938400030136108
"""
