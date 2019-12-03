# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:55:50 2019

@author: Lucinda pytorch 编程 从入门到实践 第六章-第七章
"""
alien_0={'color':'green','points':5}
print(alien_0['color'])
print(alien_0['points'])

alien={}
alien['height']=2000
alien['weight']=800
print(alien)

alien_1={'x_position':0,'y_position':25,'speed':'medium'}
print("Original x-position: "+str(alien_1['x_position']))
if alien_1['speed']=='slow':
    x_increment=1
elif alien_1['speed']=='medium':
    x_increment=2
else:
    x_increment=3
alien_1['x_position'] = alien_1['x_position']+x_increment
print("New x_position:" + str(alien_1['x_position']))

'''遍历字典中的值'''
user_0={
        'username':'efermi',
        'first':'enrico',
        'last':'fermi',}
for key, value in user_0.items():
    print("\nKey:"+key)
    print("\nValue:"+value)
    
name = input("Please enter your name:  ")
print("Hello, "+name+"!")