from numpy import int64
import pandas as pd
import matplotlib.pyplot as plt
from operator import truediv
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import numpy as np
from sklearn import metrics

df=pd.read_csv(r"sales_prediction\sales_predictions.csv")
print(df.info())
print(df.head())

#------------------------------------------EDA (START)---------------------------------------------------

df['Item_Outlet_Sales']=df['Item_Outlet_Sales'].astype('int64')
df['Item_Outlet_Sales']=df['Item_Outlet_Sales'].astype('float64')


#identify BLANKS
print('Find Blanks\n',df.isnull().sum())

print(df.groupby(['Item_Type','Item_Fat_Content']).agg({'Item_Weight': ['mean', 'min', 'max']}))
#Item weight isnt relevant for sales predication

#Identify which Outlet Size is BLANK
print(df.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'],dropna=False)['Outlet_Size'].count())

#Fill missing values of Outlet size
# df.loc[(df['Outlet_Type']=='Grocery Store') & (df['Outlet_Location_Type']=='Tier 3'),"Outlet_Size"].fillna('Small',inplace=True)
#Above code didnt work...

df.loc[(df['Outlet_Type']=='Grocery Store') & (df['Outlet_Location_Type']=='Tier 3'),"Outlet_Size"] = 'Small'
df.loc[(df['Outlet_Type']=='Supermarket Type1') & (df['Outlet_Location_Type']=='Tier 2') & (df['Outlet_Size'].isna()),"Outlet_Size"] = 'Small'

print(df.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'],dropna=False)['Outlet_Size'].count())

# Data cleanup (Renaming) 
print(df['Item_Fat_Content'].value_counts())
df.loc[df['Item_Fat_Content']=='LF','Item_Fat_Content']= 'Low Fat'
df.loc[df['Item_Fat_Content']=='low fat','Item_Fat_Content']= 'Low Fat'
df.loc[df['Item_Fat_Content']=='reg','Item_Fat_Content']= 'Regular'
print(df['Item_Fat_Content'].value_counts())

print('Top Selling Product')
print(df['Item_Type'].value_counts())
print(df['Outlet_Type'].value_counts())

#------------------------------------------EDA (END)---------------------------------------------------

#-------------------------------------------------ML (START)------------------------------------------------

# After EDA Dropping - [Item_Weight] & [Item_Identifier] - Weight bcoz of no correlation and Item_Identifier because too many Items No expands the data

df=df.drop(columns=['Item_Identifier','Item_Weight','Outlet_Identifier'])

#identify BLANKS
print(df.isnull().sum())
print(df.info())

#------------Trasforming Ordinal Categoical Values

# Grouping Date of establishment by Year with the assumption - The older the store higher the customer base
df['no_of_years']=df['Outlet_Establishment_Year'].apply(lambda x: "%.2f" % (x/10))
df['no_of_years']=df['no_of_years'].astype('float64')
df['no_of_years']=df['no_of_years'].apply(lambda x: math.trunc(x))
df['no_of_years']=df['no_of_years'].astype('int64')
print(df['no_of_years'].value_counts())
df=df.drop(columns=['Outlet_Establishment_Year'])
#### df=df.drop(columns=['Outlet_Establishment_Year','Item_Type'])

#~~Reference - Pandas OrdinalEncoder explained by TA Cassendra during ta_evals practice 
encoder = ce.OrdinalEncoder(cols=['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type','no_of_years'],return_df=True,
mapping=[{'col': 'Item_Fat_Content','mapping': {'Low Fat': 0,'Regular' : 1}},
         {'col': 'Outlet_Size','mapping': {'Small': 0,'Medium' : 1,'High' : 2}},
         {'col': 'Outlet_Location_Type','mapping': {'Tier 1': 0,'Tier 2' : 1,'Tier 3' : 2}},
         {'col': 'Outlet_Type','mapping': {'Grocery Store': 0,'Supermarket Type1' : 1,'Supermarket Type2' : 2,'Supermarket Type3' : 3}},
         {'col': 'no_of_years','mapping': {200: 0, 199 : 1,198 : 2}}])
ordinal_df = encoder.fit_transform(df)

#Trasforming Nominal Categoical Values

df_dummies = pd.get_dummies(ordinal_df,columns=['Item_Type'],drop_first=True)
print(df_dummies.head().T)
####df_dummies = ordinal_df

#Scaling
scaler = MinMaxScaler()
df_dummies[['Item_Visibility','Item_MRP']] = \
    scaler.fit_transform(df_dummies[['Item_Visibility','Item_MRP']])

# # Method 1 ------------------Linear Regression:

X = df_dummies.loc[:, df_dummies.columns !='Item_Outlet_Sales'].values
y = df_dummies.loc[:, 'Item_Outlet_Sales'].values
reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
y_result=reg.predict(X)
score = reg.score(X, y)
print('Linear Score: ',score)


# # Method 2 ------------------KNN Regression:
X=df_dummies.loc[:,df_dummies.columns !='Item_Outlet_Sales']
# print(X.shape)
y=df_dummies['Item_Outlet_Sales']
# print(y.shape)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)
predict = knn.predict(X)
score=knn.score(X,y)
print('KNN Score: ',score)

# print('KNN Model fits better than the Linear Regression ')

# Method 3 -------------------- 

# #instantiate the Random Forest Regressor Model


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

clf = RandomForestRegressor(n_estimators=500,max_depth=5,min_samples_split=3,bootstrap=True,oob_score=True)
# clf = KNeighborsRegressor(n_neighbors=5)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Random Forest Regressor Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Random Forest Regressor Test: ',score_test)


y_pred=clf.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
plt_x=list(range(len(y_test)))
print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(8,4))
sns.regplot(plt_x,'Actual')
plt.scatter(plt_x,df['Actual'],c='r')
plt.show()
# print('Random OOB Score : ', clf.oob_score_)


clf = KNeighborsRegressor(n_neighbors=5)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('KNN Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('KNN Test: ',score_test)


clf = LinearRegression(fit_intercept=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Linear Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Linear Test: ',score_test)


# Feature Importance

# def r2(rf, X_train, y_train):
#     return r2_score(y_train, rf.predict(X_train))

# perm_imp_rfpimp = permutation_importances(clf, X_train, y_train, r2)

# print(perm_imp_rfpimp)
# x_values=list(range(len(perm_imp_rfpimp)))
# plt.bar(x_values,perm_imp_rfpimp['Importance'], orientation = 'vertical')
# plt.xticks(x_values,perm_imp_rfpimp.index,rotation='vertical')
# plt.title('Feature Importance')
# plt.show()

#Grid Search

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
# clf = RandomForestRegressor(random_state=42)

# param_grid={'n_estimators': [400,500,750,1000],
#        'min_samples_split': [3,4,5,6,8,10],
#             'max_depth': [4,5,6,7,8]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=42)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)     

# cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
# cv_clf.fit(X_train,y_train)
# print('Best Parameter')
# print(cv_clf.best_params_)


#-------------------------------------------------ML (END)------------------------------------------------

#------------------------------------- #Data Visualization (START) --------------------------

# print(df['Item_Outlet_Sales'].sum())
# sales_by_Outlet = df.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum().reset_index().round(2)
# print(sales_by_Outlet['Item_Outlet_Sales'])
# count_by_Outlet = df.groupby(['Outlet_Type'])['Outlet_Identifier'].nunique()
# sales_list=list(sales_by_Outlet['Item_Outlet_Sales'].div(10000))
# Outlet_list=list(sales_by_Outlet['Outlet_Type'])
# no_of_stores=list(count_by_Outlet.values)
# avg_sales= list(map(truediv,sales_list,no_of_stores))

# for index, item in enumerate(avg_sales):
#     avg_sales[index] = int(item)
# print(avg_sales)
# result =  sales_by_Outlet['Item_Outlet_Sales'].div(count_by_Outlet.values).reset_index().round(2)

# plt.bar(Outlet_list,avg_sales)
# # plt.bar(count_by_Outlet.index,result['Item_Outlet_Sales'].values)
# plt.style.use('classic')
# plt.xlabel('Outlet Type',fontsize=10)
# plt.ylabel('Sales (In Thousands)',fontsize=10)
# plt.title('Avg by Outlet Type',c='b',fontsize=16)
# # plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# #Data Vis #2 ********Identify Top and Bottom Sale Stores

# top3=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(3)
# bottom3=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum().sort_values(ascending=False).tail(3)


# plt.figure(figsize=(16,8))
# plt.subplot(1,2,1)
# plt.bar(top3.index,top3.values,color='g')
# plt.xticks(rotation=90)
# plt.title('Top 3 Sales Store',fontsize=12)


# plt.subplot(1,2,2)
# plt.bar(bottom3.index,bottom3.values,color='r')
# plt.title('Bottom 3 Sales Store',fontsize=12)
# plt.xticks(rotation=90)
# plt.title('Top & Bottom performers')
# plt.show()

# #Data Vis 3
# # Not sure why data wiz 3 is getting mixed previous data visualizations  
# sales_by_stores=df.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum()
# sns.scatterplot(x=sales_by_stores.index,y=sales_by_stores.values)
# plt.show()

# #Data Vis 4 ********Which product sell the most based on Price
# plt.style.use('seaborn')
# df['Item_Outlet_Sales'].hist(bins=50)
# plt.ticklabel_format(useOffset=False, style='plain')
# plt.grid(False)
# plt.xticks(rotation=45)
# plt.xlabel('Price of Products')
# plt.ylabel('Vol of sales')
# plt.title('Vol. of Sales by Price')
# plt.show()

# #Data Vis 5  *********Profit by Outlet type
# plt.figure(figsize=(8,4))
# df['Profit']=df['Item_Outlet_Sales']-df['Item_MRP']
# sns.boxplot(x='Outlet_Type', y='Profit', data=df)
# plt.title('Profit by Outlet type')
# plt.xticks(rotation=45)
# plt.show()

#------------------------------------- #Data Visualization (END) --------------------------
