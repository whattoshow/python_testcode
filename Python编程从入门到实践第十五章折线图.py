# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#使用plot绘制折线图
import matplotlib.pyplot as plt
input_values=[1,2,3,4,5]#x轴的值
squares = [1,4,9,16,25]#y轴的值
plt.plot(input_values,squares,linewidth=3)
'''设置图标标题，并给坐标轴加上标签'''
plt.title("Square numbers",fontsize=24)
plt.xlabel("Value",fontsize=14)
plt.ylabel("Square of value",fontsize=14)

'''设置刻度标记的大小'''
plt.tick_params(axis='both',lablesize=14)
plt.show()


