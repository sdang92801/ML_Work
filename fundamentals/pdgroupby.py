import pandas as pd
df = pd.read_excel(r'C:\Users\Employee\Downloads\group_by.xlsx')
df=df.rename(columns={'Unnamed: 0':'Flow_ID',
                      'Unnamed: 1':'Date',
                      'Unnamed: 2':'Time',
                      'Unnamed: 3':'Count',})

date_filter = df['Date']>'2005-07-24'
flowid_filter = df['Flow_ID']=7
df2 = df.loc[date_filter & flowid_filter,:]
# df[['Date']]>'2005-07.25'
top10=df2.sort_values(by=['Count'],ascending=False).head(10)
print(top10)
top10_low = top10['Count'].min()
top10_updated = top10['Count'].apply(lambda x: x - top10_low)
print(top10_updated.mean())
data_aug=df.loc[(df['Date']>='2005-08-01') & (df['Date']<='2005-08-31'),:]
print(data_aug)    
print(data_aug.groupby(['Date']).max('Count'))