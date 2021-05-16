import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('https://query.data.world/s/d342ri3qxs5cbwtq3zh4qwfgqtpyky')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe().apply(lambda a:a.apply('{0:.2f}'.format).astype(float)))
print(df['Country'].value_counts())
print(df['Category'].value_counts())
print(df.isnull().sum())
print(df.isnull().any(axis=1))
print(df[df.isnull().any(axis=1)]['Country'].value_counts())

# print(df['Cases'].hist(bins=30))
# plt.ticklabel_format(useOffset=False, style='plain')
# plt.xticks(rotation=45)
# plt.show()

# print(sns.boxplot(x=df['Cases']))
# plt.show()


us_sales=df.loc[df['Country']=='United States',:]
us_sales['quality_col'] = us_sales['Quality'].map({'Super Premium': 'blue','Standard':'orange','Premium':'green'})
plt.scatter(us_sales['Year'],us_sales['Cases'],c=us_sales['quality_col'])
plt.show()