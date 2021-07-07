import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df=pd.read_csv(r'Assignments\Files\wisconsinBreastCancer.csv')
print(df.head())

df.drop(columns = 'Unnamed: 32', inplace = True)
X = df.drop(columns = 'diagnosis')
y = df['diagnosis']

scaler=StandardScaler()
scaled_df=scaler.fit_transform(X)
pca=PCA()
pca.fit(scaled_df)

plt.plot(range(1, 11), pca.explained_variance_ratio_[:10], marker = '.')
plt.xticks(ticks = range(1, 11))
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Explained Variance')


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)

scaler=StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

pca = PCA(n_components = 3)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

print('Training accuracy:', logreg.score(X_train_pca, y_train))
print('Testing accuracy:', logreg.score(X_test_pca, y_test))
