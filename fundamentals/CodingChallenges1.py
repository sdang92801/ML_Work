# Coding challenge 1: Digit Sum

def digit_sum():
    a= list(input('Enter a number:'))
    sum = 0
    for x in a:
        sum+=int(x)
    print(sum)
digit_sum()

# Coding challenge 2: Find Primes

# num = int(input("Enter a number: "))

# find_prime =[]
# for x in range(2,num+1):
#     # 10
#     counter = 0
#     for y in range(2,x):
#         if(x%y ==0):
#             counter = 1
#             break
#     if counter == 0:
#         find_prime.append(x)
# print(find_prime)


# Coding challenge 3: Guess Number

# import random
# num = random.randint(1,100)
# print(num)
# x = 0
# count = 0
# guess = []
# while x != num:
#     x = int(input('Enter a number between 1 to 100: '))
#     count += 1
#     if x == num:
#         print("Great!!! You guessed the correct number.")
#         print("No of tries you took to guess the correct number", count)
#     elif num > x and num - x > 5:
#         print("number is too low")
#     elif num < x and x - num > 5:
#         print("number is too high")
#     elif x in guess:
#         print('Already tried')
#     else:
#         print('Close try')
#     guess.append(x)
    

