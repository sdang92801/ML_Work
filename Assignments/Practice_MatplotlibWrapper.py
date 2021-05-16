import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Employee\Downloads\athleteEventsNoPersonal.csv")
print(df.head())
yearfilter=df['Year']== 2016
top20=df.loc[yearfilter,:].groupby(['NOC'])['Height'].mean().sort_values(ascending=False).head(20)
# plt.bar(top20.index,top20.values)
# plt.xticks(rotation=90)
# top20.plot.bar()
# plt.ylabel('Avg height in (cm)')
# plt.show()
print(df.head().T)
sns.barplot(x=top20.index,y=top20.values)
plt.xticks(rotation=90)
plt.show()

