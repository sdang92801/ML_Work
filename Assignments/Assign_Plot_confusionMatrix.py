import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


data= [[87,16],[17,59]]
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
plt.xlabel('True Label',fontsize=15)
plt.ylabel('False Label',fontsize=15)
plt.title('SNS Heatmap')
sns.heatmap(data,cmap="Blues",annot=True,annot_kws={"size": 16})
# Reference - 
# https://seaborn.pydata.org/generated/seaborn.heatmap.html 
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea - For annot 
plt.xticks(fontsize= 10)
plt.yticks(fontsize=10)
plt.show()