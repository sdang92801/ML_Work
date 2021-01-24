# #1
# def countdown(a):
#     for x in range(a,0,-1):
#         print(x)
#     print(0)
# countdown(int(input("Enter a number:")))

# #2
# def print_and_return(a):
#     print(a[0])
#     return(a[1])
# print_and_return([1,2])

# #3
# def first_plus_length(a):
#     return len(a)+a[0]
# print(first_plus_length([1,2,3,4,5]))

# #4
# def length_and_value(a,b):
#     len=[]
#     for x in range(a):
#         len.append(b)
#     return len
# print(length_and_value(4,5))

# #5
# def values_greater_than_second(a):
#     greater=[]
#     for x in range(len(a)):
#         if a[1] < a[x]:
#             greater.append(a[x])
#     if len(greater) == 0:
#         return False
#     else:
#         print(len(greater))
#         return(greater)
# print(values_greater_than_second([5,2,3,2,1,4]))