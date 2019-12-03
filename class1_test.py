# -*- coding: utf-8 -*-
"""
Spyder Editor Lucinda

This is a temporary script file.
"""
print("This is a test ")
favorite_language = 'python !'
print(favorite_language)

favorite = favorite_language.rstrip()
print(favorite)

bycicles=['trek','cannondale','redline','specialized']
print(bycicles)
print(bycicles[0])
print(bycicles[0].title())
bycicles[1]='ducati'
print(bycicles)

bycicles.append('benz')
print(bycicles)
del bycicles[4]
print(bycicles)


magicians=['alice','adam','david']
for magican in magicians:
    print(magican)
    
for value in range(1,10):
    print(value)
print(list(range(1,6)))