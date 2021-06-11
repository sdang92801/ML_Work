from Assignments.Practice_ML_Logistic_MultiClass import X_test, X_train
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve

df=pd.read_csv(r'Assignments\Files\modifiedIris2Classes.csv')
print(df.head())

X=df.drop(columns='target')
y=df['target']

logreg= LogisticRegression(C=.001)

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=3)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
logreg.fit(X_train,y_train)

print(f'Training AUC: {roc_auc_score(y_train, logreg.predict_proba(X_train)[:,1])}')
print(f'Testing AUC: {roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1])}')

plot_roc_curve(logreg, X_train, y_train)
plt.plot([0, 1], [0, 1], ls = '--', label = 'Baseline (AUC = 0.5)')
plt.legend()
