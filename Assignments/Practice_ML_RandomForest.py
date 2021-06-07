import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import category_encoders as ce
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

X=df.drop(columns=['target'])
y=df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
clf = RandomForestClassifier(n_estimators=100,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score=clf.score(X_test,y_test)
print(score)    
print(clf.oob_score_)

#Estimater range
estimator_range = [1] + list(range(10,310,10))
scores=[]

for estimator in estimator_range:
    clf=RandomForestClassifier(n_estimators=estimator,bootstrap=True,random_state=1)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_test,y_test))
    ## Why did we not predict before calculating the score.. if i use predict_score.. was the step required then?

fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
axes.plot(estimator_range,scores)
axes.set_xlabel('n_estimators',fontsize=10)
axes.set_ylabel('Accuracy',fontsize=10)
axes.grid()
plt.show()

print(scores[:5])
print(estimator_range[:5])