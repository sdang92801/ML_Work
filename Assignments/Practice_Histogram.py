import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv(r"ML_Work\Assignments\Files\kingCountyHouseData.csv")
print(df.head())
price_filter = df.loc[:, 'price'] <= 3000000
plt.style.use('seaborn')
print(df.loc[price_filter,'price'].hist(bins=30,edgecolor='black'))
plt.ticklabel_format(useOffset=False, style='plain')
plt.xticks(rotation=45)
plt.show()
