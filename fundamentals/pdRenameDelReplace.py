import pandas as pd
df = pd.read_excel(r'C:\Users\Employee\Downloads\RailsToTrails.xlsx')
print(df.head())
print(df.info())
print(df.drop(columns=['Unnamed: 6']))

df=df.rename(columns={'Unnamed: 6':'percent_change'})
print(df.head())
print(df.info())
print(df.loc[:,'2021 Counts'].fillna(0))
df=df.rename(columns={' 2019 counts (31 counters)':'counts_2019'})
print(df.head())
print(df.to_numpy())
print(df.to_dict())