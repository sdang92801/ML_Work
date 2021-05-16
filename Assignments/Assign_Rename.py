import pandas as pd
df=pd.ExcelFile(r'C:\Users\Employee\Downloads\RailsToTrails.xlsx')
bike_df=pd.read_excel(df,'Bike Counts (14 counters)')
totalCount_df=pd.read_excel(df,'Total Count data (31 counters)')
print(bike_df.info())
print(bike_df.head())

#Drop 3 columns
bike_df= bike_df.drop(columns=['Unnamed: 5','Unnamed: 6','Unnamed: 7'])

#Rename a column
bike_df=bike_df.rename(columns={'Unnamed: 8' : 'percent_change'})

print(bike_df.info())
print(bike_df.isna().sum())
print(totalCount_df.info())
totalCount_df['2021 Counts'].fillna(0)
totalCount_df['2021 Counts']=totalCount_df['2021 Counts'].fillna(0)
print(totalCount_df['2021 Counts'].head())
