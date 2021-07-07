import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.datasets import fetch_openml

# load the dataset
mnist = fetch_openml('mnist_784')
# view the shape of the dataset
print(mnist.data.shape)
X=mnist.data
y=mnist.target

scaler=StandardScaler()
scaled_df=scaler.fit_transform(X)

pca=PCA(n_components=.95)
pca.fit(scaled_df)

plt.plot(range(1, 21), pca.explained_variance_ratio_[:20], marker = '.')
plt.xticks(ticks = range(1, 21))
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Explained Variance')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

pca = PCA(n_components = 12)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#RandomForest

clf = RandomForestClassifier(n_estimators=125,max_depth=7,min_samples_split=5,bootstrap=True,oob_score=True)
clf.fit(X_train_pca,y_train)
clf.predict(X_test_pca)

print('Training accuracy:', clf.score(X_train_pca, y_train))
print('Testing accuracy:', clf.score(X_test_pca, y_test))


 # # Randomized & Grid Search
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
# clf = RandomForestClassifier(random_state=3)

# param_grid={'n_estimators': [25,50,100,125,200],
#         'min_samples_split': [5,6,7,8],
#             'max_depth': [5,6,7,8]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)

