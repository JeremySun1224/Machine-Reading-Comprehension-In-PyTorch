# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/5/22 -*-

"""
注意力机制最早是在机器视觉领域被提出的。它基于一个直观的想法，即人类在识别图像中物体时并不是观察每个像素，
而是集中注意力在图像中某些特定的部分，观察别的物体时，注意力有移动到另外的领域。

同样的道理也可以推广到文本理解中。因此，深度学习在处理文本信息时，可以根据当前的需要将权重集中到某些词语上，
而且随着时间推移，这一权重可以自动调整。这就是自然语言理解里的注意力机制。

注意力机制的输入包括两个部分：
	1、被注意的对象，为一组向量{a1, a2, ..., an}，如输入文本的词向量；
	2、一个进行注意的对象，为一个向量x。

向量x需要对{a1, a2, ..., an}进行总结，但是x对每个ai的注意力都不一样。注意力取决于从向量x的角度给被注意
对象的打分，更高的分值代表该对象更应该被关注。然后将打分用softmax归一化后并对{a1, a2, ..., an}计算加权和，
得到最终的注意力向量c。

注意力机制中的打分过程是通过注意力函数（attention function）实现的。注意力函数没有固定的形式，
只需要对两个输入向量得到一个相似度分数即可，例如使用内积函数。

向量c是{a1, a2, ..., an}的线性组合，但是权重来自x和每个ai的交互，即注意力的计算。

"""

import torch
import torch.nn.functional as F


# a：被注意的向量组，batch*m*dim
# x：需要进行注意力计算的向量组，batch*n*di
def attention(a, x):
    scores = x.bmm(a.transpose(1, 2))  # 内积计算注意力分数，结果维度为batch*n*m
    alpha = F.softmax(scores, dim=-1)  # 对最后一维进行了softmax
    attended = alpha.bmm(a)
    return attended


# 测试
batch = 10
m = 20
n = 30
dim = 15

a = torch.randn(batch, m, dim)
x = torch.randn(batch, n, dim)
res = attention(a, x)
print(res.shape)
# torch.Size([10, 30, 15])
