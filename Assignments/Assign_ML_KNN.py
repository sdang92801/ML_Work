import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.compose import ColumnTransformer


url='https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df=pd.read_csv(url)
df.columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

# --------------------EDA-------------------------
print(df.head())
print(df.isnull().sum())
print(df.info())

#--------------------ML-----------------------

# KNN regression
X=df.loc[:,['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight']]
print(X.shape)
y=df['Rings']+1.5
print(y.shape)

####Standardize Data
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X, y)
predict = knn.predict(X)
print(knn.score(X,y))

# KNN classification

X=df.loc[:,['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']]
print(X.shape)
y=df['Sex']
print(y.value_counts())

####Standardize Data
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X, y)
prediction=knn.predict(X)
print(knn.score(X,y))


