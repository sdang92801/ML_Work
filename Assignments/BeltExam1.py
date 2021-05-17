import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"ML_Work\Assignments\Files\hotel_bookings.csv")
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.head().T)

# -------------------------------- Handling Missing data and DataType (START)------------------------- 

# 1. "arrival_date_week_number" came as an object instead of Integer during df.info
print(df['arrival_date_week_number'].value_counts())
df.loc[df['arrival_date_week_number']=='#27','arrival_date_week_number']=27
df['arrival_date_week_number']=df['arrival_date_week_number'].astype('int64')


# 2. 'childern' column is a float instead of integer (becuase of missing data)
#   Handling missing data
#   There are only 4 columns where data is missing for 'children' 
#   #Assuming there are no childern- Below is the commented code, but i removed 4 lines as it will not impact analysis)
#   df.loc[:,'children'].fillna(0,inplace=True)

df.dropna(subset=['children'],inplace=True)
#   Reference - https://www.codegrepper.com/code-examples/python/drop+rows+with+nan+in+specific+column+pandas

# # Changing datatype from float64 to Int
df['children']=df['children'].astype('int64')


#3. 'agent' & 'company' have too many missing values hence can't be used for the analysis. Dropping both columns
df = df.drop(columns=['agent','company'])
print(df.info())

# -------------------------------- Handling Missing data and DataType (END) ------------------------- 

# ------------------------------------ Data Visualization (START)-------------------------

## 1. Find highest no of bookings per month 

total_booking = df.groupby(['arrival_date_month'])['is_canceled'].count().sort_values(ascending=False)
plt.style.use('seaborn')
plt.figure(figsize=(10,6))
colors=['g','b','b','b','b','b','b','b','b','b','b','r']
plt.bar(total_booking.index,total_booking.values,color=colors)
# # Reference - https://showmecode.info/matplotlib/bar/change-bar-color/

plt.xlabel('Month',fontsize=10)
plt.ylabel('No of Booking',fontsize=10)
plt.title('Bookings by Month',c='b',fontsize=16)
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

##------------------------------------
## 2. Cancellation by Lead time 

sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.xlabel('Cancellation Type',fontsize=10)
plt.ylabel('Lead Time',fontsize=10)
plt.title('Cancellation by Lead time',c='b',fontsize=16)
plt.show()

##------------------------------------
## 3. Cancellation by Deposit 

deposit = df.groupby(['deposit_type'])['is_canceled'].value_counts()
print(deposit)
cancel_by_deposit = df.loc[df['is_canceled']==1,:].groupby(['deposit_type'])['is_canceled'].count()
print(cancel_by_deposit)

plt.style.use('seaborn')
plt.figure(figsize=(6,5))
plt.bar(cancel_by_deposit.index,cancel_by_deposit.values)
plt.xlabel('Deposit Type',fontsize=10)
plt.ylabel('No of Cancellations',fontsize=10)
plt.title('Cancellations by Deposit type',c='b',fontsize=16)
plt.bar_width=15
plt.grid(False)
plt.show()

##------------------------------------
## 4. Market Segment Check

plt.figure(figsize=(10,6))
cancel_market_segment=df.loc[df['is_canceled']==1,:].groupby(['market_segment'])['market_segment'].count()
sns.scatterplot(x=cancel_market_segment.index,y=cancel_market_segment.values)
plt.xlabel('Market Segment',fontsize=10)
plt.ylabel('Number of Cancellation',fontsize=10)
plt.title('Cancellation by Market Segment',c='b',fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##------------------------------------
## 5. Further Breakdown on Online Booking Cancellation by Deposit

cancel_Online = df.loc[(df['is_canceled']==1) & (df['market_segment']=='Online TA'),:].groupby(['deposit_type'])['deposit_type'].count()
sns.scatterplot(x=cancel_Online.index,y=cancel_Online.values)
plt.xlabel('Deposit Type',fontsize=10)
plt.ylabel('Number of Cancellation',fontsize=10)
plt.title('Online Cancellation by Deposit Type',c='b',fontsize=16)
plt.show()


##------------------------------------
## 6. Customers Type providing Confirmed bookings 

cust_type=df.loc[df['is_canceled']==0,:].groupby(['customer_type'])['customer_type'].count()
plt.style.use('seaborn')
plt.figure(figsize=(6,5))
plt.bar(cust_type.index,cust_type.values)
plt.xlabel('Customer Type',fontsize=10)
plt.ylabel('No of Confirmed Bookings',fontsize=10)
plt.title('Bookings by Customer Type',c='b',fontsize=16)
plt.bar_width=15
plt.grid(False)
plt.show()