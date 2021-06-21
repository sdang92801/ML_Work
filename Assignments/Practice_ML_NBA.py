import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns





df=pd.read_csv(r'C:\Users\dangs\Desktop\Python\ML_Work\Assignments\Files\nba_rookies.csv')
print(df.head())
print(df.info())

print(df['TARGET_5Yrs'].value_counts())

df.loc[df['TARGET_5Yrs']=='No','TARGET_5Yrs']= 0
df.loc[df['TARGET_5Yrs']=='Yes','TARGET_5Yrs']= 1
df['TARGET_5Yrs']=df['TARGET_5Yrs'].astype('float64')
print(df.info())

df.drop(columns='Name',inplace=True)
X=df.drop(columns='TARGET_5Yrs')
y=df['TARGET_5Yrs']

#instantiate
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3,test_size=.80)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predict = knn.predict(X_train)
score=knn.score(X_test,y_test)
print('KNN Score: ',score)

# #RandomForest
clf = RandomForestClassifier(n_estimators=750,max_depth=4,min_samples_split=5,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Random Forest Regressor Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Random Forest Regressor Test: ',score_test)


# y_pred=clf.predict(X_test)
# df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
# plt_x=list(range(len(y_test)))
# print(df)

# #Grid Search

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
# clf = RandomForestClassifier(random_state=3)

# param_grid={'n_estimators': [400,500,750,1000],
#        'min_samples_split': [3,4,5,6,8,10],
#             'max_depth': [2,3,4,5,6]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)     

# cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
# cv_clf.fit(X_train,y_train)
# print('Best Parameter')
# print(cv_clf.best_params_)

#Logistic Regression L1
log_reg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
log_reg.fit(X_train,y_train)
print('Training Accuracy: ',log_reg.score(X_train,y_train))
print('Testing Accuracy: ',log_reg.score(X_test,y_test))

#Logistic Regression L2
log_reg = LogisticRegression(C=1,penalty='l2', solver='liblinear',multi_class='ovr')
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


# def r2(log_ab, X_train, y_train):
#     return r2_score(y_train, log_ab.predict(X_train))

# perm_imp_rfpimp = permutation_importances(log_reg, X_train, y_train, r2)

# print(perm_imp_rfpimp)
# x_values=list(range(len(perm_imp_rfpimp)))
# plt.bar(x_values,perm_imp_rfpimp['Importance'], orientation = 'vertical')
# plt.xticks(x_values,perm_imp_rfpimp.index,rotation='vertical')
# plt.title('Feature Importance')
# plt.show()


clf = DecisionTreeClassifier(max_depth = 4, random_state = 3)
clf.fit(X_train,y_train)
clf.predict(X_test)
print(clf.score(X_test,y_test))

# Finding Optimal Depth

# max_depth_length = list(range(1,6))
# accuracy = []
# for depth in max_depth_length:
#     clf=DecisionTreeClassifier(max_depth=depth,random_state=0)
#     clf.fit(X_train,y_train)
#     score = clf.score(X_test,y_test)
#     accuracy.append(score)

# plt.plot(max_depth_length,accuracy)
# plt.show()



