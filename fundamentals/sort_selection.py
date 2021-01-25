def sortselection(a):
    for run in range(len(a)):
        for x in range(len(a)):
            counter = 0
            for y in range(x,len(a)):
                if a[x]>a[y]:
                    ind=y
                    counter=1
            if counter==1:
                a[x],a[ind]=a[ind],a[x]
    print(a)
sortselection([8,5,2,6,9,3,1,4,0,7])
