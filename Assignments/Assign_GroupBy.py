import pandas as pd
url='https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/Pandas/CalIt2.data'
# Reference: https://stackoverflow.com/questions/40769691/prevent-pandas-read-csv-treating-first-row-as-header-of-column-names
df=pd.read_csv(url,header=None)
print(df.info())
print(df.head())
#Rename Columns
df.columns = ['Flow_ID','Date','Time','Count']
print(df.head())
#Selecting Data
filter_date = df['Date']=='07/24/05'
filter_flowid = df['Flow_ID']==7
top10 = df.loc[filter_date & filter_flowid,:].sort_values(by=['Count'],ascending=False).head(10)
print(top10)
#Apply function
min=top10['Count'].min()
print(min)
top10=top10['Count'].apply(lambda x:x-min)
print(top10.mean())
#Grouping
month=(df['Date']>='08/01/05') & (df['Date']<='08/31/05')
# print(df.loc[month,'Date'].value_counts())
# month_flowID= df.loc[(df['Date']>='08/01/05') & (df['Date']<='08/31/05') & (df['Flow_ID']==7),:]
# print(month_flowID.head())
# print(month_flowID.groupby(['Date'])[['Count']].max())