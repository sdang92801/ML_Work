import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


df=pd.read_csv(r"ML_Work\Assignments\Files\ta_evals.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())
print(df['score'].value_counts())

# # Change Categorical values for Ordinal Variables
# scores={'Low': 0,'Medium' : 1,'High': 2}
# df['score']=df['score'].map(scores)
# #same thing in one line of code
# # df['score']=df['score'].map({'Low': 0,'Medium' : 1,'High': 2})
# print(df.head())

encoder = ce.OrdinalEncoder(cols=['score'],return_df=True,mapping=[{'col': 'score','mapping': {'Low': 0,'Medium' : 1,'High': 2}}])
newDF = encoder.fit_transform(df)


#Nominal Categorical Variables

#1 Pandas get_dummies

df_dummies = pd.get_dummies(newDF,columns=['instructor','course','semester'],drop_first=True)
print(df_dummies.head().T)

# Sklearn OneHotEncoder
# ohc = OneHotEncoder(drop='first',sparse=False)
# ohc.fit(df[['instructor','course','semester']])
# df_ohc=ohc.transform(df[['instructor','course','semester']])
# print(df_ohc)

