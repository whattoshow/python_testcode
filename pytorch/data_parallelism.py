# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:43:48 2020

@author: Lucinda
"""
'''
###1、在本代码中，我们将会学习如何使用DataParallel来利用多个GPU进行计算
###利用pytorch使用多GPU非常方便，我们可以把一个模型放到GPU中
device = torch.device("cuda:0")
model.to(device)    
#之后，我们可以复制所有的tensor到GPU中
mytensor = my_tensor.to(device)
'''
###导入PyTorch模块以及定义参数
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#参数和数据加载器
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

#设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#虚拟数据集
#生成一个虚拟数据集，我们只需实现getitem
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

###简单的模型
###举个例子，我们的模型只获取一个输入，再执行一个线性操作，并给出输出。然而我们可以使用DataParallel再任意的模型上
###（CNN，RNN，Capsule Net 等等）
###我们已经在模型中放置了一条打印语句，以监视输入和输出张量的大小。请注意在batch rank 0什么被打印出来了
class Model(nn.Module):
    #我们的模型
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
    
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output_size())
        return output

###创建模型并使用并行数据
###这部分就是此次教程的最核心部分，首先我们需要做一个模型并检查我们是否有多个GPU，如果我们有多个GPU，那么我们可以
###使用nn.DataParallel封装模型。之后我们可以通过model.to(device)把我们的模型放在多个GPU上
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #dim = 0[30, xxx] -> [10,...],[10,...] on 3 GPUs
    model = nn.DataParallel(model)
if torch.cuda.device_count() == 1:
    print("There is only one GPU")   
model.to(device)

###训练模型，现在我们看一下输入输出的张量大小
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())





    
