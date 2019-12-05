#!/usr/bin/env python
# coding: utf-8

filename='G:\Programing\python_testcode\pi_digits.txt'
with open(filename) as file_object:
    contents = file_object.read()
    for line in contents:
        print(line)
        '''print(contents.rstrip())'''
        
with open(filename) as file_object:
    contents = file_object
    for line in contents:
        print(line)





