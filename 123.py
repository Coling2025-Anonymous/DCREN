import os

import torch
from torch import nn

# lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, bias=False, batch_first=True)
# input = torch.randn( 3,128, 768)  # batch_size为3（输入数据有3个句子），每个句子有8个单词，每个单词向量长度为10
# # h_0, c_0 = torch.randn(3, 3, 20), torch.randn(3, 3, 20)
# h_0, c_0 = torch.randn(3, 3, 20), torch.randn(3, 3, 20)
# output, (h_n, c_n) = lstm(input, (h_0, c_0))
#
#
# output, (h_n, c_n) = lstm(input)
# print(output)
#
# print('weight_ih_l0.shape = ', lstm.weight_ih_l0.shape, ', weight_hh_l0.shape = ' , lstm.weight_hh_l0.shape)
# a = nn.Linear(768, 4, bias=False)
# b = nn.Linear(768, 4, bias=False)
#
# a = a.state_dict()['weight']
# b = b.state_dict()['weight']
# a = torch.ones(8,2)
# b = None
#
# c= torch.cat((a,b),dim=-1)
# print(a)
# print(b)
# print(c)
# print(torch.cuda.is_available())

# if not os.path.exists("./checkpoint/tacred/exe"):
#     os.makedirs("./checkpoint/tacred/exe")
#  a   print("true")

a = [0,1,2]
print(a[:-1])