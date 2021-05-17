import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r"ML_Work\Assignments\Files\Week 43.xlsx")
print(df.head())
print(df.info())
print(df['Country'].value_counts())
print(df['Year'].value_counts())
export_2020=df.loc[df['Year']== 2020,:].groupby(['Country'])['Exports (USD Millions)'].sum()
export_2019=df.loc[df['Year']== 2019,:].groupby(['Country'])['Exports (USD Millions)'].sum()
plt.bar(export_2019.index,export_2019.values, label='2019')
plt.bar(export_2020.index,export_2020.values, label='2020')
plt.xlabel('Export in 2019-2020')
plt.ylabel('Export in USD Millions')
plt.title('Apparel Exports to US',c='b',fontsize=16)
plt.legend()
plt.show()
