import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

url='https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df=pd.read_csv(url)
df.columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

# --------------------EDA-------------------------
print(df.head())
print(df.isnull().sum())
print(df.info())

# ----------------------------------- ML-------------------------

#Categorical Classification - Nominal
df_dummies = pd.get_dummies(df,columns=['Sex'],drop_first=True)
print(df_dummies.head().T)

#Assign data to X & y
X=df_dummies.drop(columns=['Rings'])
y=df_dummies['Rings']
print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Instantiate Linear Regression
reg = LinearRegression(fit_intercept=True)
reg.fit(X_train,y_train)
predict_train = reg.predict(X_train)
predict_test = reg.predict(X_test)

print('Linear Regression')
print(reg.score(X_train,y_train))
print(reg.score(X_test,y_test))

#Instantiate KNN
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train,y_train)
predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

print('KNN')
print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))


# 1. Which of KNN or linear regression seemed like a better model when you didn't use train test split? 
# - My earlier model excluded 'SEX' whereas in this model i took SEX into account but KNN seems to be a better model

# 2. Which of KNN or linear regression seemed like a better model when you used train test split? 
# - Linear model gave better test results compared to KNN 

# 3. Was there an advantage to linear regression in terms the amount of code you had to write? 
# - Both KNN and linear regression have similar line of code

# 4. Is there any way you could show someone which of the two models was more effective? 
# - Comparing both the Scores of KNN and Linear will be a bettion option. Also plotting the result in a Bar graph might help visually

# Is there any way you think you could have improved KNN to be more effective of a model? 
# - The changed the KNN neighbor and imporved the score but not good enough to beat the linear score. 
# - I also feel dropping Sex column might have imporved the KNN score
