import pandas as pd
df = pd.read_csv(r'C:\Users\Employee\Downloads\mortgages.csv')
print(df.head())
df = df.rename(columns={'Starting Balance':'starting_balance',
                        'Interest Paid':'interest_paid',
                        'Principal Paid':'principal_paid'})
print(df.head())
# Approach 2
df.columns = ['Month',
              'starting_balance',
              'repayment',
              'interest_paid',
              'principal_paid',
              'new_balance',
              'mortgage_name',
              'interest_rate']
print(df.head())

# Drop columns
df = df.drop(columns=['new_balance'])
print(df.head())

