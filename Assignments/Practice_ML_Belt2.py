import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

df=pd.read_csv(r'Assignments\Files\Fish.csv')
print(df.head())
print(df.info())
df_upd=df.loc[:,['Weight','Length1','Height','Width']]
print(df_upd.corr())

# -----------------------------------EDA Start----------------------

#Heatmap
plt.figure(figsize = (8, 5))
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = 'Blues', annot = True)
# plt.show()

#Histogram for Distribution
df.loc[:,:].hist(bins=25,
                 figsize=(16,16),
                 xlabelsize='10',
                 ylabelsize='10',xrot=-15)
# plt.show()



#Scatterplot for Linear Corelaton
fig, axes = plt.subplots(nrows = 1,ncols = 3,figsize = (8,2))
sns.regplot(x='Weight', y='Length1', data=df, ci=None, ax = axes[0], scatter_kws={'alpha':0.3})
sns.regplot(x='Weight', y='Height', data=df, ci=None, ax = axes[1], scatter_kws={'alpha':0.3})
sns.regplot(x='Weight', y='Width', data=df, ci=None, ax = axes[2], scatter_kws={'alpha':0.3})
fig.tight_layout()
# plt.show()

#Correlation of different lengths
fig, axes = plt.subplots(nrows = 1,ncols = 3,figsize = (8,2))
sns.regplot(x='Weight', y='Length1', data=df, ci=None, ax = axes[0], scatter_kws={'alpha':0.3})
sns.regplot(x='Weight', y='Length2', data=df, ci=None, ax = axes[1], scatter_kws={'alpha':0.3})
sns.regplot(x='Weight', y='Length3', data=df, ci=None, ax = axes[2], scatter_kws={'alpha':0.3})
fig.tight_layout()
# plt.show()

# ----------------EDA END------------------------

# ---------------------------------------ML---------------------------

#Transform Data
df_dummies = pd.get_dummies(df,columns=['Species'],drop_first=True)
print(df_dummies.head().T)

X=df_dummies.drop(columns=['Weight'])
y=df_dummies['Weight']


#Instantiate
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=3)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# ---------Linear Start------ 
reg = LinearRegression(fit_intercept=True)
reg.fit(X_train,y_train)
y_result=reg.predict(X_test)
score = reg.score(X_test, y_test)
print('Linear Refression: ',score)

# ---------Linear Ends------

#--------- Random Forest Start------

# -------------------

# Grid Search model was run and code was commented after finding the best parameters

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3)
# clf = RandomForestRegressor(random_state=42)

# param_grid={'n_estimators': [400,500,750,1000],
#        'min_samples_split': [2,3,4,5],
#             'max_depth': [4,5,6,7,8]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)  

# # ---------------------------------

clf = RandomForestRegressor(n_estimators=750,max_depth=5,min_samples_split=3,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Random Forest')
print('Random Forest Regressor Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Random Forest Regressor Test: ',score_test)


y_pred=clf.predict(X_test)
# df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
# plt_x=list(range(len(y_test)))
# print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# plt.figure(figsize=(8,4))
# sns.regplot(plt_x,'Actual')
# plt.scatter(plt_x,df['Actual'],c='r')
# plt.show()
# print('Random OOB Score : ', clf.oob_score_)

#--------- Random Forest End------

#--------- KNN Start------

clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('KNN')
print('KNN Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('KNN Test: ',score_test)

y_pred=clf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#--------- KNN End------

#--------- Decision Tree Start------

clf = DecisionTreeRegressor(max_depth = 4, random_state = 0)
clf.fit(X_train,y_train)
clf.predict(X_test)
print('DecisionTree Score',clf.score(X_test,y_test))


# Finding Optimal Depth

# After finding Optimal Length code was commented
# max_depth_length = list(range(1,6))
# accuracy = []
# for depth in max_depth_length:
#     clf=DecisionTreeRegressor(max_depth=depth,random_state=0)
#     clf.fit(X_train,y_train)
#     score = clf.score(X_test,y_test)
#     accuracy.append(score)

# plt.plot(max_depth_length,accuracy)
# plt.show()

#--------- Decision Tree End------
