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