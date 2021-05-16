import pandas as pd
import matplotlib.pyplot as plt
from operator import truediv
import seaborn as sns
df=pd.read_csv(r"C:\Users\Employee\Downloads\sales_predictions.csv")
print(df.info())
print(df.head())
df['Item_Outlet_Sales']=df['Item_Outlet_Sales'].astype('int64')

#identify BLANKS
print(df.isnull().sum())

print(df.groupby(['Item_Type','Item_Fat_Content']).agg({'Item_Weight': ['mean', 'min', 'max']}))
#Item weight isnt relevant for sales predication

#Identify which Outlet Size is BLANK
print(df.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'],dropna=False)['Outlet_Size'].count())

#Fill missing values of Outlet size
# df.loc[(df['Outlet_Type']=='Grocery Store') & (df['Outlet_Location_Type']=='Tier 3'),"Outlet_Size"].fillna('Small',inplace=True)
#Above code didnt work...

df.loc[(df['Outlet_Type']=='Grocery Store') & (df['Outlet_Location_Type']=='Tier 3'),"Outlet_Size"] = 'Small'
df.loc[(df['Outlet_Type']=='Supermarket Type1') & (df['Outlet_Location_Type']=='Tier 2') & (df['Outlet_Size'].isna()),"Outlet_Size"] = 'Small'

print(df.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'],dropna=False)['Outlet_Size'].count())

# Data cleanup (Renaming) 
print(df['Item_Fat_Content'].value_counts())
df.loc[df['Item_Fat_Content']=='LF','Item_Fat_Content']= 'Low Fat'
df.loc[df['Item_Fat_Content']=='low fat','Item_Fat_Content']= 'Low Fat'
df.loc[df['Item_Fat_Content']=='reg','Item_Fat_Content']= 'Regular'
print(df['Item_Fat_Content'].value_counts())

print(df['Item_Type'].value_counts())
print(df['Outlet_Type'].value_counts())


# #Data Visualization
# print(df['Item_Outlet_Sales'].sum())
# sales_by_Outlet = df.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum().reset_index().round(2)
# print(sales_by_Outlet['Item_Outlet_Sales'])
# count_by_Outlet = df.groupby(['Outlet_Type'])['Outlet_Identifier'].nunique()
# sales_list=list(sales_by_Outlet['Item_Outlet_Sales'].div(10000))
# Outlet_list=list(sales_by_Outlet['Outlet_Type'])
# no_of_stores=list(count_by_Outlet.values)
# avg_sales= list(map(truediv,sales_list,no_of_stores))

# for index, item in enumerate(avg_sales):
#     avg_sales[index] = int(item)
# print(avg_sales)
# result =  sales_by_Outlet['Item_Outlet_Sales'].div(count_by_Outlet.values).reset_index().round(2)

# plt.bar(Outlet_list,avg_sales)
# # plt.bar(count_by_Outlet.index,result['Item_Outlet_Sales'].values)
# plt.style.use('classic')
# plt.xlabel('Outlet Type',fontsize=10)
# plt.ylabel('Sales (In Thousands)',fontsize=10)
# plt.title('Avg by Outlet Type',c='b',fontsize=16)
# # plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# #Data Vis #2

# top3=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(3)
# bottom3=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum().sort_values(ascending=False).tail(3)


# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plt.bar(top3.index,top3.values,color='g')
# plt.xticks(rotation=90)
# plt.title('Top 3 Sales Store',fontsize=12)


# plt.subplot(1,2,2)
# plt.bar(bottom3.index,bottom3.values,color='r')
# plt.title('Bottom 3 Sales Store',fontsize=12)
# plt.xticks(rotation=90)
# plt.title('Top & Bottom performers')

# #Data Vis 3
# # Not sure why data wiz 3 is getting mixed previous data visualizations  
# sales_by_stores=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum()
# sns.scatterplot(x=sales_by_stores.index,y=sales_by_stores.values)
# plt.show()

#Data Vis 4
plt.style.use('seaborn')
df['Item_Outlet_Sales'].hist(bins=50)
plt.ticklabel_format(useOffset=False, style='plain')
plt.grid(False)
plt.xticks(rotation=45)
plt.xlabel('Price of Products')
plt.ylabel('Vol of sales')
plt.title('Vol. of Sales by Price')
plt.show()

#Data Vis 5
plt.figure(figsize=(10,6))
df['Profit']=df['Item_Outlet_Sales']-df['Item_MRP']
sns.boxplot(x='Outlet_Type', y='Profit', data=df)
plt.title('Profit by Outlet type')
plt.xticks(rotation=45)
plt.show()