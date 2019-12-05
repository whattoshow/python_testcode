# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class Dog():
    '''一次模拟小狗的简单尝试'''
    
    def __init__(self,name,age):
        self.name=name
        self.age=age
        
    def sit(self):
        print(self.name.title()+" is now sitting.")
        
    def roll_over(self):
        print(self.name.title()+" rolled over!")
        
my_dog=Dog('willie',6)
print("My dog's name is"+ my_dog.name.title()+", its age is "+str(my_dog.age)+"~")
my_dog.sit()
my_dog.roll_over()