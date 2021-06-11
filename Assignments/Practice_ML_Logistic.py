import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r'Assignments\Files\wisconsinBreastCancer.csv')
print(df.head())

plt.scatter(df['concave points_worst'],df['diagnosis'])
plt.ylabel('malignant (1) or benign (0)', fontsize = 12)
plt.xlabel('concave points_worst', fontsize = 12)
# plt.show()

X=df[['concave points_worst']]
print(X.shape)
y=df['diagnosis']

# Linear Regression

lr= LinearRegression()
lr.fit(X,y)
prediction=lr.predict(X)
plt.scatter(df['concave points_worst'],df['diagnosis'])
plt.plot(df['concave points_worst'],prediction,color='r')
plt.ylabel('malignant (1) or benign (0)', fontsize = 12)
plt.xlabel('concave points_worst', fontsize = 12)
plt.show()

# Logistic Regression
logreg = LogisticRegression(C=1000)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
logreg.fit(X_train,y_train)

example_df = pd.DataFrame(data = {'worst_concave_points': X_test.flatten(),
                     'diagnosis': y_test})
example_df['logistic_preds'] = pd.DataFrame(logreg.predict_proba(X_test)).loc[:, 1].values
example_df = example_df.sort_values(['logistic_preds'])

plt.scatter(example_df['worst_concave_points'],example_df['diagnosis'])
plt.plot(example_df['worst_concave_points'],example_df['logistic_preds'],color='r')
plt.ylabel('malignant (1) or benign (0)', fontsize = 12)
plt.xlabel('concave points_worst', fontsize = 12)
plt.show()

