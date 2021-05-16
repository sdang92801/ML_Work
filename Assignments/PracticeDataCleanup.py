import pandas as pd
df=pd.read_csv(r'C:\Users\Employee\Downloads\super_bowl.csv')
print(df.info())
print(df.head())
print(df.shape)
# - Drop columns
df.drop(columns='Side Judge',inplace=True)
# - Code to drop all rows that have NA
#   df.dropna(inplace=True)

# - Drop column where every single col of the row is blank
#   df.dropna(how='any',inplace=True)
# - Check for duplicates and delete duplicate records
print(df.duplicated().any())
# df.drop_duplicated(inplace=True)
df.drop(columns=['Referee','Umpire','Head Linesman','Line Judge','Field Judge','Back Judge'],inplace=True)

# - Change data type
print(df.dtypes)
df['Date']=pd.to_datetime(df['Date'])

print(df['State'].value_counts())
df.loc[df['State']=='FL','State'] = 'Florida'
print(df['Losing Pts'])
print(df.isnull().sum())

# dp.drop(columns='Unnamed: 0',inplace = True)

#- Outliers in Numerical column
# from scipy import stats

# z = np.abs(stats.zscore(df['Attendance']))
# print(np.where(z > 3))
