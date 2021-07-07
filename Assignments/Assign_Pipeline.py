import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

url='https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df=pd.read_csv(url)
df.columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


X= df.drop(columns='Rings')
y= df['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# Create a pipeline for scaling, PCA, & logistic regression
pipe = make_pipeline(StandardScaler(),LinearRegression(fit_intercept=True))
pipe.fit(X_train, y_train)

print('Linear Training accuracy:', pipe.score(X_train, y_train))
print('Linear Testing accuracy:', pipe.score(X_test, y_test))

#--------------------------

X= df.drop(columns='Sex')
y= df['Sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# Create a pipeline for scaling, PCA, & logistic regression
pipe = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
pipe.fit(X_train, y_train)

print('KNN Training accuracy:', pipe.score(X_train, y_train))
print('KNN Testing accuracy:', pipe.score(X_test, y_test))

#Why would you want to use a pipeline for KNN?
    # less code .. no need for multiple variable assignment
#What other models or tasks would a pipeline be useful for?
    # All of the data model where lot of Hypertuning is not required