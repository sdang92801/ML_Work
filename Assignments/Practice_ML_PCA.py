from os import scandir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df=pd.read_csv(url,names = ['sepal length','sepal width','petal length','petal width','target'])
print(df.head())

X=df.drop(columns='target')
le=LabelEncoder()
y=le.fit_transform(df['target'])

scaler=StandardScaler()
scaled_df=scaler.fit_transform(X)

pca=PCA(n_components=2)
pcs=pca.fit_transform(scaled_df)


plt.figure(figsize = (8, 4))
plt.scatter(pcs[:,0], pcs[:,1], c = y)
plt.title('Visualization of all of our data using the first two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()