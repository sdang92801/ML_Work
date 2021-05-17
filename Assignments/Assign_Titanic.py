import pandas as pd
df=pd.read_csv(r'ML_Work\Assignments\Files\titanic.csv')
print(df.head())
print(df.dtypes)
#Survival %
print(df.loc[df['Survived']==1,'Survived'].sum()/df['Survived'].count())
#Survival Count by Sex
print(df.loc[df['Survived']==1,'Sex'].value_counts())
#Survival% among ppl with Fare<10
print(df.loc[(df['Survived']==1) & (df['Fare']<10),'Survived'].sum()/df.loc[df['Fare']<10,'Survived'].count())
#Avg age of ppl who didnt survive
print(df.loc[df['Survived']==0,'Age'].mean())
#Avg age of ppl who did survive
print(df.loc[df['Survived']==1,'Age'].mean())
#Avg Age of ppl who survived n didnt survive by Sex
print(df.groupby(['Sex'])[['Age']].mean())
