import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from rfpimp import permutation_importances
from sklearn import metrics
from sklearn.metrics import r2_score

url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
col_name=['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
        'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

df=pd.read_csv(url,header=None,names=col_name)
print(df.head())
print(df['Class'].value_counts())

X=df.loc[:,df.columns!='Class'].values
y=df.loc[:,'Class'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3,stratify=y,train_size=.3)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


#Logistic Regression L1
log_reg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
log_reg.fit(X_train,y_train)
print('Training Accuracy: ',log_reg.score(X_train,y_train))
print('Testing Accuracy: ',log_reg.score(X_test,y_test))

# Feature Importance
# get importance
# print(log_reg.intercept_)
# importance = log_reg.coef_[0]
# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()


def r2(log_ab, X_train, y_train):
    return r2_score(y_train, log_ab.predict(X_train))

perm_imp_rfpimp = permutation_importances(log_reg, X_train, y_train, r2)

print(perm_imp_rfpimp)
x_values=list(range(len(perm_imp_rfpimp)))
plt.bar(x_values,perm_imp_rfpimp['Importance'], orientation = 'vertical')
plt.xticks(x_values,perm_imp_rfpimp.index,rotation='vertical')
plt.title('Feature Importance')
plt.show()


# #Logistic Regression L2
# log_reg = LogisticRegression(C=1,penalty='l2', solver='sag',multi_class='ovr')
# log_reg.fit(X_train,y_train)
# print('Training Accuracy: ',log_reg.score(X_train,y_train))
# print('Testing Accuracy: ',log_reg.score(X_test,y_test))

# #KNN Classifier
# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train,y_train)
# knn.predict(X_test)
# score_train=knn.score(X_train,y_train)
# print('KNN Train: ',score_train)    
# score_test=knn.score(X_test,y_test)
# print('KNN Test: ',score_test)

# #Grid Search
# # -------------------


# # X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
# # clf = RandomForestClassifier(random_state=3)

# # param_grid={'n_estimators': [900,1000,1100,1200],
# #        'min_samples_split': [9,10,11,12,13],
# #             'max_depth': [3,4,5,6,7]}

# # cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=42)
# # cv_rs.fit(X_train,y_train)
# # print('Random Search : ',cv_rs.best_params_)  
# # ---------------------------------

# #Random Forest

# clf = RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_split=10,bootstrap=True,oob_score=True)
# clf.fit(X_train,y_train)
# clf.predict(X_test)
# score_train=clf.score(X_train,y_train)
# print('Random Forest Regressor Train: ',score_train)    
# score_test=clf.score(X_test,y_test)
# print('Random Forest Regressor Test: ',score_test)

# #Logistic Regression L1 gave the highest Accuracy Score of 96.8% followed by Random Forest @ 95.2% and KNN @ 94.4%
# # Proline & Total phenols are the most important features 
