import pandas as pd
import numpy as np
import category_encoders as ce 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

data=load_iris()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

X = df.drop(columns=['target'])
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)

clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train,y_train)
clf.predict(X_test)
print(clf.score(X_test,y_test))

# Finding Optimal Depth

max_depth_length = list(range(1,6))
accuracy = []
for depth in max_depth_length:
    clf=DecisionTreeClassifier(max_depth=depth,random_state=0)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    accuracy.append(score)

plt.plot(max_depth_length,accuracy)
plt.show()





