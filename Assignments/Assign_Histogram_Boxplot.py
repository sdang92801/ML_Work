import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_excel(r"C:\Users\Employee\Downloads\catsvdogs.xlsx")
print(df.head())
print(df.info())
df['Percentage of Dog Owners'].hist(bins=30,label='Dogs',edgecolor='black')
df['Percentage of Cat Owners'].hist(bins=30,label='Cats',edgecolor='green')

#3. Observation - What can we see by comparing these two histograms? What information does this tell us?
# 25-30% of the total Animal Owners have cats whereas 35-45% of the people are Dog owners
plt.grid(None)
plt.title('Percentage of Animal Owners')
plt.legend()
plt.show()


mean_dogs = df['Mean Number of Dogs per household'].values
print(mean_dogs)
mean_dogs = mean_dogs[~np.isnan(mean_dogs)]
mean_cats = df['Mean Number of Cats'].values
mean_cats = mean_cats[~np.isnan(mean_cats)]
print(mean_cats)
plt.boxplot([mean_dogs,mean_cats],labels=['Dogs','Cats'])
plt.title('Mean Number of Animals')
plt.legend()
plt.show()
#5. Observation - What can we see by comparing these two boxplots? What information does this tell us?
# Interquartile range of Dogs is from 1.4-1.7 
# Whereas the Interquartile range of Cats is higher from 1.9 to 2.2
