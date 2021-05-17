import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv(r"ML_Work\Assignments\Files\modifiedBostonHousing.csv")
print(df.head())
df = df.loc[:, ['RM', 'LSTAT','PTRATIO', 'price']]
print(df.head())
print(df.isnull().sum())
df.dropna(how='any',inplace=True)
print(df.isnull().sum())
print(df.corr().sort_values(by=['price']))
