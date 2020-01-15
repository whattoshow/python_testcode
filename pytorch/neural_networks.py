# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:47:23 2020

@author: Lucinda
"""

###使用torch.nn包可以建立神经网络，我们现在已经对autograd有了初步的认识，nn依靠autograd来
###定义模型并区分它们。一个nn.Module包含多个层，使用方法“forward(input)”可以返回"output"
###卷积神经网络就是一个简单的前馈网络。它接收输入，并将输入的数据投入到各个层，最后给出结果。
###一个典型的神经网络训练过程如下：
###1、为神经网络定义一些可以学习的参数（或权重）
###2、遍历一个输入数据集
###3、在网络中处理输入数据
###4、计算损失（就是计算输出结果和正确答案差多少）
###5、反向传播梯度到网络参数中
###6、更新网络中的权重，通常使用一个简单的更新规则：weight = weight - learning_rate * gradient

'''首先定义网络'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #一个输入图像通道，六个输出通道，3*3尺寸的卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)#第一层
        self.conv2 = nn.Conv2d(6, 16, 3)#第二层，第一层的六个输出通道输入到第二层（即第二层有6个输入通道），定义16个输出通道，卷积核仍为3*3的
        #一个放射操作：y = Wx + b
        #CLASS torch.nn.Linear(in_features, out_features, bias=True)该方法用来计算线性函数
        '''Parameters:
            in_features – size of each input sample
            out_features – size of each output sample
            bias – If set to False, the layer will not learn an additive bias. Default: True'''
        self.fc1 = nn.Linear(16 * 6 * 6, 120)#6*6来自于图像的尺寸
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
        
    def forward(self, x):
        #最大池化在一个(2, 2)尺寸的窗口上
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)) 
        #如果尺寸是一个方形，那么你只可以指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]#所有尺寸除了batch 尺寸（每次投喂的图像尺寸）
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print(net)

###到此为止，我们已经定义好了前向传播函数，反向传播功能在你使用autograd的时候被自动定义了。你可以使用任意的tensor操作在前向传播函数中。
###模型的可学习参数可以通过net.parameters()返回
params = list(net.parameters())
print(len(params))
print(params[0].size())

###尝试一个随机的32*32的输入。记住：除了LeNet的输入尺寸为32x32，为了使用这个网络在MNIST数据集上，请将图像转换成32X32这一尺寸。
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

###注意torch.nn只支持mini-baches，整个torch.nn包只支持样本的mini-batch输入，而不是一个单一的样本。
###例如，nn.Conve2d带着4维的张量，包含nSamples * nChannels * Height * Width.
###如果你有一个单一的例子，用input.qunsqueeze(0)就可以添加一个假的batch维度。
###torch.Tensor 是一个多维数组，它支持自动梯度计算，例如backward()。另外也包括梯度w.r.t张量。
###nn.Module是一个神经网络模块。一个简便的封装参数的方式，可方便的将参数移动到GPU，导出、加载中。
###nn.Parameter 是一种张量，当作为一个模块的属性时，它将被自动的注册。
### autograd.Function 它实现自动梯度计算的前馈和反向传播的定义。每个Tensor操作至少创造一个Function节点，这个节点链接功能并创造一个Tensor以及将这个张量的历史做编码。

###损失函数
###损失函数将（输出结果，目标）对作为输入，计算输出结果和目标之间的差距是多少
###在nn包里有几种损失函数。一个简单的损失函数即：nn.MSELoss，这个函数计算输入和目标之间的均方差误差

###例如：
output = net(input)
target = torch.randn(10) #这个是个假目标例子
target = target.view(1, -1) #使这个假目标和输出的形状一样
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

###现在，如果你在反向传播方向依照loss这个结果，使用loss的.grad_fn属性，你会看见一个计算图，如下所示：
###input->conv2d->relu->maxpool2d->relu->maxpool2d
###     ->view->linear->relu->linear->relu->linear
###     ->MSELoss
###     ->loss

###为了描述的更清楚，我们可以看一些反向传播的步骤
print(loss.grad_fn) #MSELoss


###在使用神经网络时，为了使用使用不同的更新规则，我们可以用torch.optim来实现，
#创建自己的优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

#训练过程中用优化器对损失函数的反向传播优化:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
