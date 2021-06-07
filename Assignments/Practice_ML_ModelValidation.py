import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

X=df.drop(columns='target')
print(X.shape)
y=df['target']


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Instantiate
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
prediction = knn.predict(X_train)
predict_test = knn.predict(X_test)

score_train=knn.score(X_train,y_train)
score_test=knn.score(X_test,y_test)
print(score_train)
print(score_test)


