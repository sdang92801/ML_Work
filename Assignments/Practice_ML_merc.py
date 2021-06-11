from numpy import int64
import pandas as pd
import matplotlib.pyplot as plt
from operator import truediv
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r'Assignments\Files\merc.csv')
print(df.head())
print(df.dtypes)
print(df['model'].value_counts())
print(df['transmission'].value_counts())

df=df.drop(columns=['tax'])
df = df.loc[~(df['engineSize']==0),:]
df = df.loc[~(df['transmission']=='Other'),:]
df = df.loc[~(df['fuelType']=='Other'),:]

print(df.dtypes)

# df = df.loc[~df['engineSize']==0,:] 
# print(df['engineSize'].value_counts())

df_dummies = pd.get_dummies(df,columns=['model','transmission','fuelType'],drop_first=True)
print(df_dummies.head().T)


X = df_dummies.loc[:,df_dummies.columns !='price'].values
y= df_dummies.loc[:,'price'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# -------------------


# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
# clf = RandomForestRegressor(random_state=42)

# param_grid={'n_estimators': [400,500,750,1000],
#        'min_samples_split': [3,4,5,6,8,10],
#             'max_depth': [4,5,6,7,8]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=42)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)  
# ---------------------------------

clf = RandomForestRegressor(n_estimators=1000,max_depth=8,min_samples_split=5,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Random Forest Regressor Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Random Forest Regressor Test: ',score_test)


y_pred=clf.predict(X_test)
# df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
# plt_x=list(range(len(y_test)))
# print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# plt.figure(figsize=(8,4))
# sns.regplot(plt_x,'Actual')
# plt.scatter(plt_x,df['Actual'],c='r')
# plt.show()
# print('Random OOB Score : ', clf.oob_score_)


clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('KNN Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('KNN Test: ',score_test)

y_pred=clf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Linear Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Linear Test: ',score_test)

