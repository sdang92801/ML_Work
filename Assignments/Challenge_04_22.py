# def calculate (a,b):
#     print (a+b)
#     print (a-b)
#     print(a*b)

# calculate(3,4)

# def check(a,b):
#     print(f"Hello {a} {b}! Welcome to the wonderful world of Python!")

# check("Supreet","Dang")

def numsum(a):
    val=0
    for x in a:
        val+=int(x)
    print(val)
a = input('Enter a number')
numsum(a)

# a=0
# while a<=1:
#     a = int(input('Enter a number'))
#     if a>0:
#         numsum(a)
