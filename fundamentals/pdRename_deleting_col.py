# import pandas as pd
# df = pd.read_csv(r'C:\Users\Employee\Downloads\mortgages.csv')
# print(df.head())
# df = df.rename(columns={'Starting Balance':'starting_balance',
#                         'Interest Paid':'interest_paid',
#                         'Principal Paid':'principal_paid'})
# print(df.head())
# # Approach 2
# df.columns = ['Month',
#               'starting_balance',
#               'repayment',
#               'interest_paid',
#               'principal_paid',
#               'new_balance',
#               'mortgage_name',
#               'interest_rate']
# print(df.head())

# # Drop columns
# df = df.drop(columns=['new_balance'])
# print(df.head())

# mortgage_filter = df['mortgage_name']=='30 Year'
# interest_filter = df['interest_rate']==0.03

# df[mortgage_filter].to_csv(path_or_buf='oneMortgage.csv',index=False)
# df[interest_filter].to_excel(excel_writer='oneMortgage.xlsx',index=False)

import pandas as pd 
df = pd.read_excel(r'C:\Users\Employee\Downloads\TMo.xlsx')
print(df.head())
print(df.info())