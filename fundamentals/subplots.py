import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\Employee\Downloads\linearWithWithout.csv')
df.head()

interceptfilter = df['intercept']==True
df_intercept = df.loc[interceptfilter,:]
df_nointercept = df.loc[~interceptfilter,:]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1);
plt.plot(df_intercept['feature'],df_intercept['predicted'],c='r');
plt.scatter(df_intercept['feature'],df_intercept['actual'],c='k');
plt.title('Intercept',fontsize=13);

plt.subplot(1,2,2);
plt.plot(df_nointercept['feature'],df_nointercept['predicted'],c='r');
plt.scatter(df_nointercept['feature'],df_nointercept['actual'],c='k');
plt.title('No Intercept',fontsize=13);

plt.legend()
plt.show()