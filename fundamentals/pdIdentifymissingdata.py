import pandas as pd
df = pd.read_csv(r'C:\Users\Employee\Downloads\linear.csv')
print(df.head())
print(df.info())
print(df['y'].isna().head())
y_missing = df['y'].isna()
print(df.loc[~y_missing,:])
print(df['y'].isna().sum())