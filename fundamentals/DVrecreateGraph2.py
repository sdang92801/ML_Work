import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel(r'C:\Users\Employee\Downloads\IncomeByState.xlsx')
print(df.head())
Alambama_state = df[(df['State']=="Alabama") & (df['Year']==2016)]
Income = Alambama_state['Income Level']
Numberofhouse = Alambama_state['Number of Households']
plt.style.use('dark_background')
plt.bar(Income,Numberofhouse,label='2019 Alabama')
plt.xlabel('Income Level')
plt.xticks(rotation=45)
plt.ylabel('No. of Households')
plt.title(label='2009 Alabama Households by Income Level',fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()
