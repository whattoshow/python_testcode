# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:27:00 2019

@author: Lucinda
"""

#使用scatter()绘制散点图并设置样式
import matplotlib.pyplot as plt
x_values=[1,2,3,4,5]
y_values=[1,4,9,16,25]
plt.scatter(x_values,y_values,s=200)#2,4是点的坐标位置，s是点的大小

#设置图标的标题并给坐标轴加上标签
plt.title("Square Numbers",fontsize=24)
plt.xlabel("Value",fontsize=14)
plt.ylabel("Square of Value",fontsize=14)

#设置刻度标记的大小
plt.tick_params(axis='both',which='major',labelsize=14)
plt.show()