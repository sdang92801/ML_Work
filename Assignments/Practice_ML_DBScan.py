import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import DBSCAN

df=pd.read_csv(r"Assignments\Files\modified_wine.csv")
df=df[['malic_acid', 'flavanoids']]
print(df.head())

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

#Instantiate the model
dbs= DBSCAN(eps=0.5, min_samples=5).fit(scaled_df)
df['clusters']=dbs.labels_
plt.scatter(df['malic_acid'],df['flavanoids'],c=df['clusters'])
plt.xlabel('Malic Acid')
plt.ylabel('Flavanoids')
plt.title('Cluster of Wine Varieties')
plt.show()

