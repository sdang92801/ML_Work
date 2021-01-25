def bubbleSort(a):
    for x in range(len(a)):
        for x in range(0,len(a)-1):
            if a[x]>a[x+1]:
                a[x],a[x+1]=a[x+1],a[x]
    return a
print(bubbleSort([1,5,3,2,0,8]))

