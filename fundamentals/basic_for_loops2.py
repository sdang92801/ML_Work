# #1
# def bigger(a):
#     for x in range(0,len(a)):
#         if a[x]>0:
#             a[x]='big'
#     return a

# print(bigger([-1, 3, 5, -5]))

# #2
# def count_positive(a):
#     cnt = 0
#     for x in a:
#         if x >0:
#             cnt +=1
#     a[-1]= cnt
#     return a
# print(count_positive([-1,1,1,1]))

# # 3
# def sum_total(a):
#     total = 0
#     for x in a:
#         total += x
#     return total
# print(sum_total([1,2,3,4]))

# #4
# def avg(a):
#     length = len(a)
#     total = 0
#     for x in a:
#         total += x
#     return total / length
# print(avg([1,2,3,4]))

# #5
# def length(a):
#     return len(a)
# print(length([37,2,1,-9]))

# #6
# def minimum(a):
#     a.sort()
#     return(a[0])
# print(minimum([37,2,1,-9]))

# #7
# def maximum(a):
#     a.sort()
#     return(a[-1])
# print(maximum([37,2,1,-9]))

# #8
# def ultimate_analysis(a):
#     a.sort()
#     return({'sumTotal': sum(a),'average': sum(a)/len(a),'minimum':a[0],'maximum':a[-1],'length': len(a)})
# print(ultimate_analysis([37,2,1,-9]))

# 9
def reverse_list(a):
    return(a[::-1])
print(reverse_list([37,2,1,-9])) 
