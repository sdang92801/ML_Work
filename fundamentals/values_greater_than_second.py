def values_greater_than_second(a):
    greater=[]
    for x in range(len(a)):
        if a[1] < a[x]:
            greater.append(a[x])
    if len(greater) == 0:
        return False
    else:
        print(len(greater))
        return(greater)
print(values_greater_than_second([5,2,3,2,1,4]))