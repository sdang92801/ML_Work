import pandas as pd
df=pd.ExcelFile(r'C:\Users\Employee\Downloads\RailsToTrails.xlsx')
bike_df=pd.read_excel(df,'Bike Counts (14 counters)')
print(bike_df.info())
print(bike_df.head())
#Select just the data after July 31, 2020.
greater_July31=bike_df['Week of']>'2020-07-31'
print(bike_df.loc[greater_July31,:]) 

#Select just data where the 'Change 2019-2020' column is greater than 100% (greater than 1).
greater_100=bike_df['Change 2019-2020']>1
print(bike_df.loc[greater_100,:])

#Put both filters together
print(bike_df.loc[greater_July31 & greater_100,:])