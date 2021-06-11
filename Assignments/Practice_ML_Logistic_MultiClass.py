from operator import mul
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.tools.datetimes import Scalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
col_names = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']
df=pd.read_csv(url,header=None,names=col_names)
print(df.head().T)
print(np.unique(df['Class label']))
print(df['Class label'].value_counts())

X = df.loc[:,df.columns !='Class label'].values
y= df.loc[:,'Class label'].values

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
unique, counts = np.unique(y_train,return_counts=True)
print(dict(zip(unique,counts)))

unique, counts = np.unique(y_test,return_counts=True)
print(dict(zip(unique,counts)))

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

lreg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
lreg.fit(X_train,y_train)
print('Training Accuracy: ',lreg.score(X_train,y_train))
print('Testing Accuracy: ',lreg.score(X_test,y_test))

print(lreg.intercept_)
print(lreg.coef_)
print(lreg.predict_proba(X_test[0:1]))
print(lreg.predict(X_test[0:1]))