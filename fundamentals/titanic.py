import pandas as pd
df = pd.read_csv(r'C:\Users\Employee\Downloads\titanic.csv')
print(df.head())
survive = df['Survived']==1
print(survive.mean())
print(df.loc[survive,:].groupby(['Sex']).sum())
survive_fare=df['Fare']<10
print(survive_fare.mean())
print(df.loc[~survive,:]['Age'].mean())
print(df.loc[survive,:]['Age'].mean())
print(df.groupby(['Sex']).mean())