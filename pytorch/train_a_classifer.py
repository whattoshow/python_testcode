# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:39:24 2020

@author: Lucinda
"""

'''PyTorch官方教程'''
###1、使用torchvision，这个库对于加载CIFAR10非常轻松
import torch
import torchvision
import torchvision.transforms as transforms

###输出的torchvision数据集是PILImage图像序列[0,1]。
###我们把这些数据集转换成标准化序列的Tensors[-1, 1]...
###需要注意的是，如果在Windows系统运行，我们会得到一个BrokePipeError的提示
###可以尝试设置torch.utils.data.DataLoader()的num_worker为0

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#windows运行时，num_workder如果填写的是2，则会出现BrokenPipeError错误，在windows下填为0可以继续执行，原理待查
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###2、以下代码展示了一些训练图像
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 #非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


#获取一些随机训练图像
dataiter = iter(trainloader)
images, labels = dataiter.next()

#显示图像
imshow(torchvision.utils.make_grid(images))
#打印标签
print(''.join('%5s' % classes[labels[j]] for j in range(4)))

###3、定义卷积神经网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()

####4、定义损失函数和优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    
###5、训练网络
###这是一个比较有趣的过程，我们只需要遍历数据迭代器，然后将数据送到网络并进行优化
for epoch in range(2): # loop over the dataset multiple times 在数据及上迭代循环很多次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0 ):
        #获取输入：数据是一个[inputs, labels]的列表
        inputs, labels = data
        #梯度参数使零
        optimizer.zero_grad()
        #前馈+反向+优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #打印分析
        running_loss += loss.item()
        if i % 2000 == 1999: #每2000次迭代mini-batch打印一次
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0
            
print('Finished Training')

###6、保存我们训练好的模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH) 

###7、测试网络
###我们已经在训练数据及上训练了两次网络，但是我们需要检查网络是否学到了什么
###我们通过预测神经网络输出的类别标签对比实际情况来进行检查，如果预测的是正确，就将样本添加到正确预测列表中。
###第一步，显示一下测试集中的图像
dataiter = iter(testloader)
images, labels = dataiter.next()

#打印图像
imshow(torchvision.utils.make_grid(images))
print('GroundTruch: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))    
#接下来，让我们重新加载保存的模型
net = Net()
net.load_state_dict(torch.load(PATH)) 
#接下来，我们可以看看神经网络如何看待以上示例
outputs = net(images)
#输出的是10个类别的相似度，预测结果与某一个类别的相似度越高，那么这个网络会觉得这个图像越属于这个类别
#一下是最高相似度的索引
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
    
###接下来，我们看一下网络在整个数据集的表现
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))    
###那么接下来，我们可以看看那些分类表现良好，哪些分类表现的不好：
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        
for i in range(10):
    print('Accurary of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))
###8、在GPU上进行训练
### 就像我们将Tensor传输到GPU上，我们也可以将神经网络传输到GPU上。
### 首先让我们来定义我们的设备为第一可见cuda的设备。

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#如果我们正在一个CUDA机器上运行，那么这将会打印出一个CUDA设备
print(device)
    
#这部分我们说的设备是指CUDA设备，然后，这些方法将递归遍历所有模块，并将其参数和缓冲区转换为CUDA张量：
net.to(device)
#另外，需要将每一步的输入和输出目标也都发送到GPU中：
inputs, labels = data[0].to(device), data[1].to(device)    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    