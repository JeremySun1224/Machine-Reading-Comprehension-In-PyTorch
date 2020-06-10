# -*- coding: utf-8 -*-
"""BertTest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i8adiiJIW7qSFMl1f4NEfwd2n-sc3SgU
"""

!nvidia-smi

# 安装BERT在内的Transformer软件包

!pip install pytorch-transformers

import torch
from pytorch_transformers import *

# 使用BERT-base模型，不区分大小写

config = BertConfig.from_pretrained('bert-base-uncased')

# BERT使用的分词工具

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 载入为区间答案型阅读理解任务预设的模型，包含前向网络输出层

model = BertForQuestionAnswering(config=config)

# 处理训练数据

# 获得文本分词后的单词编号，维度为batch_size*seq_length

input_ids = torch.tensor(tokenizer.encode('This is an example')).unsqueeze(0)

# 标准答案在文本中的起始位置和终止位置，维度为batch_size

start_positions = torch.tensor([1]); start_positions

end_positions = torch.tensor([3]); end_positions

# 获得模型的输出结果

outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

outputs

# 得到交叉熵损失函数值loss，以及模型预测答案在每个位置开始和结束的打分start_scores与end_scores，维度均为batch_size*seq_length

loss, start_scores, end_scores = outputs

print('Loss：{loss}'.format(loss=loss))
print('Loss：{start_scores}'.format(start_scores=start_scores))
print('Loss：{end_scores}'.format(end_scores=end_scores))