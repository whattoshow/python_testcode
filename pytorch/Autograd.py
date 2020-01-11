# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:17:31 2020

@author: Lucinda
"""
#%%
###autograd包提供了所有张量的自动微分功能
###创建一个张量x，并且设置它的requires_grad属性为True，这样系统就会开始追踪这个张量的所有操作
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x+2
###因为y是计算的结果，所以在运行结果中，我们可以看到它有一个grad_fn属性
print(y)
z = y * y * 3
out = z.mean()
print(z,out)
###.requires_grad_(...)会改变张量的requires_grad属性。输入的标志默认为False
a = torch.randn(2, 2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###反向传播
out.backward()
###打印梯度d(out)/dx
print(x.grad)

###接下来我们可以看一个向量-雅克比成绩的实例
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
###在这个例子里，y不再是标量，.torch.autograd 不会直接计算全部的雅克比矩阵，但是如果我们想
###要得到向量-雅克比乘积，只需要传递backward参数给向量。
v = torch.tensor([0.1, 1.0, 0.00001], dtype=torch.float)
y.backward(v)
print(x.grad)

###同样我们也可以停止追踪tensor的历史，
####利用.requires_grad=True，通过将这一代码块包裹在with torch.no_grad()中
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
    
###也可以通过使用.detach()来获取一个新的tensor，该tensor具有相同的内容但是却没有要求梯度。
print(x.requires_grad)
y = x.detach()###y就是新的tensor
print(y.requires_grad)
print(x.eq(y).all())