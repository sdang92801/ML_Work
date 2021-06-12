import pandas as pd
import matplotlib.pyplot as py
df=pd.read_csv(r"ML_Work\Assignments\Files\athleteEventsNoPersonal.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
no_height=df['Height'].isnull()
df=df.loc[~no_height,:]
print(df.isnull().sum())
print(df.groupby(['ID'])[['Height']].mean())


####