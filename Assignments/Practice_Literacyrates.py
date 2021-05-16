import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Employee\Downloads\literacy_rates.csv")
print(df.info())
print(df.shape)
print(df.head())
print(df.isnull().sum())

#Filter missing value
print(df[df['Region'].isna()])
df['Region'].fillna(method = "ffill",inplace=True)
print(df.isnull().sum())
df.loc[df['Literacy rate'] == "45.384%",['Literacy rate']] = .45384
#Change Data Type
df['Literacy rate'] = df['Literacy rate'].astype(float)
print(df.info())
df.groupby['Country']['Literacy rate'].mean()


