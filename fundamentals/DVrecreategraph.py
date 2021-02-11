import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\Employee\Downloads\mortgages.csv')
ThreePercent = df.loc[(df['Interest Rate'] == 0.03) & (df['Mortgage Name'] == '30 Year'),:]
FivePercent = df.loc[(df['Interest Rate'] == 0.03) & (df['Mortgage Name'] == '30 Year'),:]
print(df.loc[ThreePercent,:])
# Three_index = list(df.loc[ThreePercent,'Month'].values)
# Three_value = list(df.loc[ThreePercent,:]['Interest Paid'].cumsum().values)
# Five_index = list(df.loc[FivePercent,:]['Month'].values)
# Five_value = list(df.loc[FivePercent,:]['Interest Paid'].cumsum().values)

# plt.plot(Three_index,Three_value)
# plt.plot(Five_index,Five_value)   
# plt.show()