class User():
    def __init__(self,name,email):
        self.name = name
        self.email = email
        self.account_balance = 0
    def deposit(self,amount):
        self.account_balance +=amount
    def withdrawal(self,amount):
        self.account_balance -=amount
    def balance(self):
        print(self.account_balance)
    # def money_transfer(self,transfer_to,amount):
    #     self.account_balance -= amount
    #     transfer_to.account_balance += amount
    #     print(self.account_balance)
    #     print(transfer_to.balance)

sam = User('Sam Dang',"samd@gmail.com")
geoff = User('Geoff Lasky','glasky@gmail.com')
kevin = User('Kevin Mac','kmac@gmail.com')
sam.deposit(100)
sam.deposit(50)
sam.deposit(10)
print(sam.account_balance)
geoff.deposit(250)
geoff.deposit(100)
geoff.withdrawal(50)
geoff.withdrawal(50)
print(geoff.account_balance)
kevin.deposit(100)
kevin.withdrawal(10)
kevin.withdrawal(20)
kevin.withdrawal(5)
print(kevin.account_balance)
# sam.money_transfer(User.kevin,50)


