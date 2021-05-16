import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\Employee\Downloads\mortgages.csv')
print(df.head())
print(df.info())
print(df['Interest Rate'].value_counts())
print(df['Mortgage Name'].value_counts())

mortgage_30yr_3percent=(df['Mortgage Name']=='30 Year') & (df['Interest Rate']==0.03)
mortgage_30yr_5percent=(df['Mortgage Name']=='30 Year') & (df['Interest Rate']==0.05)

x_3per=df.loc[mortgage_30yr_3percent,'Month'].values
y_3per=df.loc[mortgage_30yr_3percent,'Interest Paid'].cumsum().values

x_5per=df.loc[mortgage_30yr_5percent,'Month'].values
y_5per=df.loc[mortgage_30yr_5percent,'Interest Paid'].cumsum().values
print(x_3per,y_3per)
plt.style.use('classic')
plt.plot(x_3per,y_3per,c='k',label='3 Percent')
plt.plot(x_5per,y_5per,c='b',label='5 Percent')
plt.xlabel('Months',fontsize=15)
plt.ylabel('Dollars',fontsize=15)
plt.xticks(fontsize= 10)
plt.yticks(fontsize=10)
plt.title('Interest Rate in 30 yrs')
plt.legend(loc='lower right')
# plt.legend(loc=(1.02,0))
plt.tight_layout()
plt.grid(c='g',axis='y',linestyle='-')
# plt.savefig('MATLABlegendcutoff.png', dpi = 300)
plt.show()