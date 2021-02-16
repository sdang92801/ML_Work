import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\Employee\Downloads\mortgages.csv')
ThreePercent = df.loc[(df['Interest Rate'] == 0.03) & (df['Mortgage Name'] == '30 Year'),:]
FivePercent = df.loc[(df['Interest Rate'] == 0.05) & (df['Mortgage Name'] == '30 Year'),:]
Three_index = list(ThreePercent['Month'].values)
Three_value = list(ThreePercent['Interest Paid'].cumsum().values)
Five_index = list(FivePercent['Month'].values)
Five_value = list(FivePercent['Interest Paid'].cumsum().values)
print(Three_value)
plt.plot(Three_index,Three_value,c='k',label="30 Years 3% APR")
plt.plot(Five_index,Five_value,c='b',label="30 Years 5% APR")
plt.xlabel('Months')
plt.ylabel('Dollars')
plt.title('Cumalative Interest Paid in 30 yrs')
plt.legend(loc="lower right")
plt.show()