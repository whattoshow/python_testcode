# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:04:01 2019

@author: Lucinda Python编程 从入门到实践 第八章

"""
'''def 函数名(参数一,参数二,……)'''
'''定义函数'''
def describe_pet(animal_type, pet_name):
    """显示宠物的信息"""
    print("\nI have a "+animal_type+".")
    print("\nMy "+animal_type+"'s name is "+pet_name.title()+".")
'''调用函数'''
describe_pet('dog','WangZai')
describe_pet('cat','XiongMmao')

describe_pet(animal_type='Rabbit',pet_name='Alice')

"""返回值函数"""
def get_formatted_name(first_name, last_name):
    """返回整洁的姓名"""
    full_name = first_name+" "+last_name
    return full_name.title()
musician = get_formatted_name("jimi",'hendrix')
print(musician)

"""让实参编程可选的,可选的参数后面要给个空值"""
def get_formatted_name(first_name,last_name,middle_name=''):
     if middle_name:
        full_name=first_name+" "+middle_name+" "+last_name
    else:
        full_name=first_name+" "+last_name
    return full_name
musician=get_formatted_name('jimi','hendrix')
print(musician)
musician=get_formatted_name('jimi','hooker','hendrix')
print(musician)