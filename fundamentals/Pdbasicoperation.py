import pandas as pd 
df = pd.read_excel(r'C:\Users\Employee\Desktop\EmployeeData.xlsx')
# print(df.head(2))
# print(df.tail(2))
# print(df.dtypes)
# print(df.info())
# print(df.shape)
# print(df[['Type']].head())
# print(df[['Type','Status']].head())
# print(df['Status'].head())
# print(df['Status'][0:2])
# print(df.loc[:,'Status'].head())
print(df['Type'].value_counts())
Act_filter = df['Type'] == 'Activation'
Act = df[Act_filter]
print(Act)
print(df.loc[df['Type']=='Activation','Comment'])
Status_filter = df['Status'] == 'Reported'
print(df.loc[Act_filter & Status_filter, :] )
