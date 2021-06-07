import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df=pd.read_csv(r"ML_Work\Assignments\Files\modifiedBostonHousing.csv")
print(df.head())
df = df.loc[:, ['RM', 'LSTAT','PTRATIO', 'price']]
print(df.head())
print(df.isnull().sum())
df.dropna(how='any',inplace=True)
print(df.isnull().sum())
print(df.corr().sort_values(by=['price']))
price_filter=df.loc[:,'price']<0
df=df.loc[~price_filter,:]
print(df.loc[:,:].hist(bins=25,
                 figsize=(16,16),
                 xlabelsize='10',
                 ylabelsize='10',xrot=-15))

fig, axes = plt.subplots(nrows = 1,ncols = 3,figsize = (10,2))
sns.regplot(x='RM', y='price', data=df, ci=None, ax = axes[0], scatter_kws={'alpha':0.3});
sns.regplot(x='LSTAT', y='price', data=df, ci=None, ax = axes[1], scatter_kws={'alpha':0.3});
sns.regplot(x='PTRATIO', y='price', data=df, ci=None, ax = axes[2], scatter_kws={'alpha':0.3});
fig.tight_layout()
# plt.show()

X = df.loc[:, ['RM', 'LSTAT', 'PTRATIO']].values
y = df.loc[:, 'price'].values
reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
print(reg.predict(X[0].reshape(-1,3)))
print(reg.predict(X[0:10]))
score = reg.score(X, y)
print(score)

# Mean Absolute Error
y_pred= reg.predict(X)
print(mean_absolute_error(y, y_pred))

# Mean Squared Error
print(mean_squared_error(y, y_pred))

#Root Mean Absolute Error
print(mean_squared_error(y, y_pred,squared=False))