import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\Employee\Downloads\archive\athlete_events.csv')
print(df.head())

# No of unique olympians per year
getuniqueyear = df.groupby(['Year'])['ID'].nunique()
print(getuniqueyear)

#Convert to numpy array or list
getYear = np.array(getuniqueyear.index)
getOlympians = np.array(getuniqueyear.values)
print(getYear)
print(getOlympians)

plt.style.use('dark_background')
plt.plot(getYear,getOlympians,c='magenta')
plt.grid(axis='y')
plt.xlim(left=1890,right=2020)
plt.ylim(bottom=0,top=12000)
plt.xlabel('Year')
plt.ylabel('No of Olympians')
plt.title('Olympians by Year',fontsize = 20)
plt.legend()
plt.show()

