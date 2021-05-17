import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"ML_Work\Assignments\Files\linearWithWithout.csv")
print(df.head())
intercept_filter=df['intercept']==True
df_intercept=df.loc[intercept_filter,:]
df_nointercept=df.loc[~intercept_filter,:]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(df_intercept['feature'].values,df_intercept['predicted'].values,c='r')
plt.scatter(df_intercept['feature'].values,df_intercept['actual'].values,c='k')
plt.title('intercept',fontsize=12)

plt.subplot(1,2,2)
plt.plot(df_nointercept['feature'].values,df_nointercept['predicted'].values,c='r')
plt.scatter(df_nointercept['feature'].values,df_nointercept['actual'].values,c='k')
plt.title('No intercept',fontsize=12)
plt.show()