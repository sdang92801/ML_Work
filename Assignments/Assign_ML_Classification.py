import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.datasets import make_classification


df=pd.read_csv(r'Assignments\Files\bank_modified.csv')
print(df.head())

X=df.drop(columns='y_yes')
y=df['y_yes']

log_reg= LogisticRegression(C=.001)

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=3,stratify=y)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
log_reg.fit(X_train,y_train)

# print(f'Training AUC Score : {roc_auc_score(y_train, log_reg.predict_proba(X_train)[:,1])}')
# print(f'Testing AUC Score : {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])}')


#Logistic Regression L1
# log_reg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
# log_reg.fit(X_train,y_train)
# print('Training Accuracy Score L1: ',log_reg.score(X_train,y_train))
# print('Testing Accuracy Score L1: ',log_reg.score(X_test,y_test))
# print(f'Training AUC L1: {roc_auc_score(y_train, log_reg.predict_proba(X_train)[:,1])}')
# print(f'Testing AUC L1: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])}')


# Logistic Regression L2
log_reg = LogisticRegression(C=10,penalty='l2', solver='sag',multi_class='ovr')
log_reg.fit(X_train,y_train)
print('Training Accuracy Score L2: ',log_reg.score(X_train,y_train))
print('Testing Accuracy Score L2: ',log_reg.score(X_test,y_test))
print(f'Training AUC Score L2: {roc_auc_score(y_train, log_reg.predict_proba(X_train)[:,1])}')
print(f'Testing AUC Score L2: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])}')

plot_roc_curve(log_reg, X_train, y_train)
plt.plot([0, 1], [0, 1], ls = '--', label = 'Baseline (AUC = 0.5)')
plt.legend()
# plt.show()

y_pred=log_reg.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
print('Specificity: ',specificity)

print('Precision: %.3f' % precision_score(y_test, y_pred))

# X, y = make_classification(
#     n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
#     n_features=20, n_samples=1000, random_state=10
# )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)

# clf = LogisticRegression(class_weight="balanced")
# clf.fit(X_train, y_train)

THRESHOLD = 0.50
preds = np.where(log_reg.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)

output=pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds), roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"])

print(output)
print("TN: ", tn," FN: ",fn,"\nFP: ",fp," TP: ",tp)