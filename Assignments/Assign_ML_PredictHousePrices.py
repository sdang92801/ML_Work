import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df=pd.read_csv(r'ML_Work\Assignments\Files\kc_house_data.csv')
print(df.head())
print(df.isnull().sum())
print(df.info())
df['date']=df['date'].str.rstrip('T000000').astype('object')
print(df['date'])


X=df.drop(columns=['price'])
y=df['price']


#instantiate the Random Forest Regressor Model

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2)
clf = RandomForestRegressor(n_estimators=100,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score=clf.score(X_test,y_test)
print('Random Forest Regressor')
print(score)    
print(clf.oob_score_)

# Feature Importance
importance = clf.feature_importances_
print(importance)
plt.barh(X.columns,importance)
plt.show()


# instantiate BaggingRegressor model

clf = BaggingRegressor(n_estimators=100,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score=clf.score(X_test,y_test)
print('Bagging Regressor')
print(score)    
print(clf.oob_score_)

#Grid Search

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
clf = RandomForestRegressor(random_state=42)

param_grid={'n_estimators': [200,500],
            'max_depth': [4,5,6,7,8]}

cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
cv_clf.fit(X_train,y_train)
print(cv_clf.best_params_)

#1. What are the most important features for your model? 
# Grade and Sqrt Living are the most important features

#2. What other parameters could you have tried tuning? 
# Used Grid Search to play around with Parameters
# Will Try RandomizedSearchCV in one of the future assignments 



# ---------------- Dont Run - Its going out of Memory ------------ 
# 
# estimator_range = [1] + list(range(10, 310, 30))
# scores = []
# for estimator in estimator_range:
#     clf = RandomForestClassifier(n_estimators=estimator,
#                                  random_state=1,
#                                  bootstrap=True)
#     clf.fit(X_train, y_train)
#     scores.append(clf.score(X_test, y_test))

# print(estimator_range)
# print(scores)

# plt.plot(estimator_range,scores)
# plt.show()

# ---------------- Dont Run - Its going out of Memory ------------ 
