import random
def randInt(min='',max=''):
    if min=='' and max=='': 
        return round(random.random()*100)
    elif min !='' and max =='':
        return round(random.random() *(100-min) + min)
    elif min =='' and max !='':
        return round(random.random() * max)
    else:
        return round(random.random() * (max-min) + min)
print(randInt())
print(randInt(max=50))
print(randInt(min=50))
print(randInt(min=50, max=500))