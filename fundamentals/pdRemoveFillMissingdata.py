import pandas as pd 
df = pd.read_csv(r'C:\Users\Employee\Downloads\linear.csv')
print(df)
print(df.info())
print(df.loc[0:10,:].dropna(how='any'))
print(df.loc[0:10,'y'].fillna(0))
print(df.loc[0:10,'y'].fillna(method='bfill'))
print(df.loc[0:10,'y'].fillna(method='ffill'))
print(df.loc[0:10,'y'].interpolate(method='linear'))