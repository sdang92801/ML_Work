def insertionSort(a):
    for x in range(len(a)):
        for y in range(x,0,-1):
            if a[y]<a[y-1]:
                a[y],a[y-1]=a[y-1],a[y]
    print(a)
insertionSort([6,5,8,1,8,7,2,4])