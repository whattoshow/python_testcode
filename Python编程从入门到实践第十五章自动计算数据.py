# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:33:29 2019

@author: Lucinda
"""

#自动计算数据
import matplotlib.pyplot as plt

x_values=list(range(1,100))
y_values=[x**2 for x in x_values]
plt.figure(dpi=100,figsize=(5,4))
plt.scatter(x_values,y_values,c=(1,0,0),edgecolor='none',s=10)#c=(红，绿，蓝),edgecolor='none'表示不描边

plt.xlabel("Value of X",fontsize=14 )
plt.ylabel("Value of Y",fontsize=14)
plt.axis=([0,1100,0,110000])

plt.savefig("square_plot.png",bbox_inches='tight')#保存图像
plt.show()
