import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score

df=pd.read_csv(r"Assignments\Files\modified_wine.csv")
df=df[['malic_acid', 'flavanoids']]
print(df.head())

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

silhouette_scores= []
for i in range(2,11):
    kmeans= KMeans(i)
    kmeans.fit(scaled_df)
    silhouette_scores.append(silhouette_score(scaled_df,kmeans.labels_))

plt.plot(range(2,11),silhouette_scores,marker='.')
plt.xlabel('No of Clusters')
plt.ylabel('Silhouette Score')
# plt.show()

kmeans = KMeans(n_clusters = 2)
kmeans.fit(scaled_df)
silhouette_score(scaled_df, kmeans.labels_)

hc = AgglomerativeClustering(n_clusters = 2)
hc.fit(scaled_df)
print(silhouette_score(scaled_df, hc.labels_))

dbs = DBSCAN(eps = 0.5, min_samples = 5).fit(scaled_df)
print(silhouette_score(scaled_df, dbs.labels_))
