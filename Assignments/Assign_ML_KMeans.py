import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cust_info=pd.read_csv(r'Assignments\Files\cust_seg.csv')
print(cust_info.head())
print(cust_info.info())
df=cust_info[['Age','Years Employed','Income','Card Debt','Defaulted','DebtIncomeRatio']]
df=df.dropna(how='any')
print(df.head())
print(df.info())

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

kmeans=KMeans(n_clusters=3)
kmeans.fit(scaled_df)
df['cluster'] = kmeans.labels_
# plt.scatter(df['Age'],df['Years Employed'],df['Defaulted'],df['DebtIncomeRatio'])
plt.figure(figsize=(16,8))
plt.subplot(1,3,1)
plt.scatter(df['Defaulted'], df['Years Employed'], c = df['cluster'])
plt.xlabel('Defaulted')
plt.ylabel('Year Employed')
plt.subplot(1,3,2)
plt.scatter(df['Defaulted'], df['Age'], c = df['cluster'])
plt.xlabel('Defaulted')
plt.ylabel('Age')
plt.subplot(1,3,3)
plt.scatter(df['Defaulted'], df['DebtIncomeRatio'], c = df['cluster'])
plt.xlabel('Defaulted')
plt.ylabel('DebtIncomeRatio')
plt.show()

#What are the trends in your segments?
#   Longer the employee less the default ratio
#   Default ratio is lower as long as Debit Income Ratio is below 25
