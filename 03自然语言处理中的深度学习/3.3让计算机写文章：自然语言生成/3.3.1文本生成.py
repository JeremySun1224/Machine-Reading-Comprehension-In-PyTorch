# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/5/21 -*-

"""
生成文本的方式是将状态转化为单词的过程，所以这一网络结构被称为解码器（decoder）。需要注意的
是，解码器的使用必须从做到右的单向RNN，而不能使用双向RNN，因为解码的过程是依次产生下一个词。
使用双向RNN会导致模型在训练时“偷看”后面的词从而达到近100%的准确率，但这样训练出来的参数是
没有意义的。


由于神经网络初始化采用随机生成的参数，所以在训练初期解码器的生成效果可能很差，这就会导致最开
始生成的单词是错的。由于生成的单词会作为输入进入下一个RNN模块，就会造成错误累积。因此，在训
练解码器的时候，一个常用的技巧是teacher forcing（强制教学）。teacher forcing策略在训练
时并不会将解码器生成的单词输入到下一个RNN模块，而是直接使用真实答案输入到RNN模块的。这样可
以有效缓解解码器训练困难的问题，参数训练难度小很多，易于更快地收敛得到高质量的解码器。实际应
用中也可以采用teacher forcing和非teacher forcing交叉使用的策略。
"""

# 使用teacher forcing策略的NLGNet模型
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class CNNMaxPooling(nn.Module):
    def __init__(self, word_dim, window_size, out_channels):  # out_channels表示CNN输出通道数
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


class NLGNet(nn.Module):
    # word_dim为词向量长度，window_size为CNN窗口长度，rnn_dim为RNN的状态维度，vocab_size为词汇表大小
    def __init__(self, word_dim, window_size, rnn_dim, vocab_size):
        super(NLGNet, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)  # 单词编号与词向量对应参数矩阵，nn.Embedding为PyTorch自带词向量参数矩阵
        self.cnn_max_pooling = CNNMaxPooling(word_dim=word_dim, window_size=window_size, out_channels=rnn_dim)
        self.rnn = nn.GRU(word_dim, rnn_dim, batch_first=True)
        self.linear = nn.Linear(in_features=rnn_dim, out_features=vocab_size)

    # x_id：输入文本的词编号，维度为batch*x_seq_len
    # y_id：真值输出文本的词编号，维度为batch*y_seq_len
    # 输出预测的每个位置每个单词的得分word_scores，维度是batch*y_seq_len*vocab_size
    def forward(self, x_id, y_id):
        x = self.embed(x_id)  # 得到输入文本的词向量，维度为batch*x_seq_len*word_dim
        y = self.embed(y_id)  # 得到真值输出文本的词向量，维度为batch*y_seq_len*word_dim
        doc_embed = self.cnn_max_pooling(x)  # 输入文本向量，结果维度是batch*cnn_channels
        h0 = doc_embed.unsqueeze(0)  # 输入文本向量作为RNN的初始状态，结果维度为1*batch*y_seq_len*rnn_dim
        rnn_output, _ = self.rnn(y, h0)  # teacher forcing。RNN后得到每个位置的状态，结果维度是batch*y_seq_len*rnn_dim
        word_scores = self.linear(rnn_output)  # 每一个位置所有单词的分数，结果维度是batch*y_seq_len*vocab_size
        return word_scores


vocab_size = 100  # 100个单词
net = NLGNet(word_dim=30, window_size=10, rnn_dim=15, vocab_size=vocab_size)
# 共30个输入文本的词id，每个文本长度为10
x_id = torch.LongTensor(30, 10).random_(0, vocab_size)
# 共30个真值输出文本的词id，每个文本长度为8
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=1)
word_scores = net(x_id, y_id)  # 每个位置词表中每个单词的得分word_scores，维度为30*8*vocab_size
loss_func = nn.CrossEntropyLoss()
# 将word_scores变为二维数组，y_id变为一维数组，计算损失函数
loss = loss_func(word_scores[:, : -1, :].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss1 = {loss1}'.format(loss1=loss))
optimizer.zero_grad()
loss.backward()
optimizer.step()
word_scores = net(x_id, y_id)
loss = loss_func(word_scores[:, : -1, :].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))  # reshape也可改为view()
print('loss2 = {loss2}'.format(loss2=loss))

"""
loss1 = 4.629878044128418
loss2 = 4.572292327880859
"""
