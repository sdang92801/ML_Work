from os import X_OK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error



mpg=pd.read_csv(r"ML_Work\Assignments\Files\auto-mpg.csv")
print(mpg.head())

print(mpg['origin'].value_counts())
print(mpg['model year'].nunique())
mpg = pd.get_dummies(mpg, columns = ['origin', 'model year'], drop_first = True)
print(mpg['car name'].nunique())
mpg.drop(columns = 'car name', inplace = True)
X=mpg.drop(columns=['mpg'])
y=mpg['mpg']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

print('Mean: ',y_train.mean())

train_preds=[y_train.mean()]*len(y_train)
test_preds=[y_test.mean()]*len(y_test)

print('Baseline Train: ',np.sqrt(mean_squared_error(y_train,train_preds)))
print('Baseline Test: ',np.sqrt(mean_squared_error(y_test,test_preds)))

knn=KNeighborsRegressor(n_neighbors=len(X_train))
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print('KNN Training: ',np.sqrt(mean_squared_error(y_train,train_preds)))
print('KNN Test: ', np.sqrt(mean_squared_error(y_test,test_preds)))


knn=KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print('KNN Overfit Training: ',np.sqrt(mean_squared_error(y_train,train_preds)))
print('KNN Overfit Test: ',np.sqrt(mean_squared_error(y_test,test_preds)))

#Balance


Test=[]
Neigh=[]
for knn_nei in range(1,16):
    knn=KNeighborsRegressor(n_neighbors=knn_nei)
    knn.fit(X_train, y_train)
    Neigh.append(knn_nei)
    train_preds = knn.predict(X_train)
    test_preds = knn.predict(X_test)
    Test.append(np.sqrt(mean_squared_error(y_test,test_preds)))
    print('KNN Overfit Training for {}: '.format(knn_nei),np.sqrt(mean_squared_error(y_train,train_preds)))
    print('KNN Overfit Test for {}: '.format(knn_nei),np.sqrt(mean_squared_error(y_test,test_preds)))
    print('-------------------------')

print(Neigh)
print(Test)
plt.plot(Neigh,Test)
plt.show()