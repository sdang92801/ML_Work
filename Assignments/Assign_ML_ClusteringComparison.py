import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score

df=pd.read_csv(r"C:\Users\dangs\Desktop\Python\ML_Work\Assignments\Files\cust_seg.csv")
print(df.head())
print(df.info())
df=df.dropna(how='any')
print(df.info())

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

silhouette_scores=[]

for i in range(2,12):
    kmean=KMeans(i)
    kmean.fit(scaled_df)
    silhouette_scores.append(silhouette_score(scaled_df,kmean.labels_))


plt.plot(range(2,12),silhouette_scores,marker='.')
plt.xlabel('No of clusters')
plt.ylabel('Silhouette Score')
# plt.show()

kmean=KMeans(n_clusters=2)
kmean.fit(scaled_df)
print('KMean Silhouette Score: ',silhouette_score(scaled_df, kmean.labels_))

hc = AgglomerativeClustering(n_clusters = 2)
hc.fit(scaled_df)
print('Agglomerative Silhouette Score: ',silhouette_score(scaled_df, hc.labels_))

dbs = DBSCAN(eps = 0.5, min_samples = 3).fit(scaled_df)
print('DBSCAN Silhouette Score: ',silhouette_score(scaled_df, dbs.labels_))
# print(dbs.labels_)

#I kept chaning the DBScan eps (radius). it took the entire dataset under 1 umbrella 
# Otherwise KMean gives better results than Agglomerative or DBScan approach
