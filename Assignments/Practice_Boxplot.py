#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df=pd.read_csv(url,header=None)
print(df.shape)
df.columns=['id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
print(df.head())
malignant = df.loc[df['diagnosis']=='M','area_mean'].values
benign = df.loc[df['diagnosis']=='B','area_mean'].values
# plt.boxplot([malignant,benign], labels=['M', 'B'])
# df.boxplot(column='area_mean',by='diagnosis')
sns.boxplot(x='diagnosis', y='area_mean', data=df)
plt.show()




# %%
