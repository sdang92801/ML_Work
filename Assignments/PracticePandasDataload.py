# import pandas as pd
# df= pd.read_excel(r'ML_Work\Assignments\Files\bostonHousing1978.xlsx')
# print(df.info())
# print(df[['RM','LSTAT']][0:10])
# print(df.loc[0:10,['RM','LSTAT']])

import pandas as pd
df=pd.read_csv(r'ML_Work\Assignments\Files\mortgages.csv')
print(df.info())
print(df['Mortgage Name'].value_counts())
print(df['Interest Rate'].value_counts())
mort_fil=df['Mortgage Name']=='30 Year'
int_fil=df['Interest Rate']==0.03
print(df.loc[mort_fil & int_fil,:])

# @@ Try .loc with selective columns
# @@ print(df.loc[mort_fil,['Starting Balance','Interest Paid']].head())