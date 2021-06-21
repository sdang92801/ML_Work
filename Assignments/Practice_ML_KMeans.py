import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

wine=pd.read_csv(r'Assignments\Files\modified_wine.csv')
df = wine[['malic_acid', 'flavanoids']]
print(df.head())
scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

plt.scatter(df['malic_acid'], df['flavanoids'])
plt.xlabel('Malic Acid')
plt.ylabel('Flavanoids')
# plt.show()

kmeans=KMeans(n_clusters=2)
kmeans.fit(scaled_df)
df['cluster'] = kmeans.labels_
plt.scatter(df['malic_acid'], df['flavanoids'], c = df['cluster'])
plt.xlabel('Malic Acid')
plt.ylabel('Flavanoids')
plt.title('Clusters of Wine Varieties')
plt.show()