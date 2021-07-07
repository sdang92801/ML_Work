import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import category_encoders as ce
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns



df=pd.read_excel(r"C:\Users\dangs\Downloads\Arch_Employee_Data (3).xlsx")
print(df.head())
print(df.info())

#----------------------------------------- Data Cleaning [START]-------------------------

#Deleting data that has lot of missing data
df=df.drop(columns=['Work_Location','Position_Code','Employment_Predictor_Score','Act._Marital_Status',
                        'Exempt_Status','Length_of_Service_Since_Rehire','Position_Family','Business_Title','#State_Exemptions/Allowances',
                        'Block_State_Tax?','Estimated_Deductions','State_Filing_Status','State_Filing_Status_Desc','W2_Info_12DD_(2020)',
                        'RehirefromT-Mobile','ESS_EEO1_Ethnicity/Race','Supervisor_Approval','Supervisor_Approval_Code','Budgeted',
                        'Supervisor_Primary','Supervisor_Primary_Code','Supervisor_Secondary','Supervisor_Secondary_Code','Supervisor_Tertiary',
                        'Supervisor_Tertiary_Code','TerminationCategory','TrainingLocation','Current_Key_Employee','ESS_Language_Preference',
                        'EEO1_Category','Termination_Reason','Termination_Type','Business_Title.1','Achieved & GP','Last Name','First Name',
                        'd','Fixed Name','Month1.1','MOnth2','Month3.1','Month4.1','Month5.1','Month6.1','Name','Type'])

#Deleting data that doesnt make any relevance with Attrition
df.drop(columns=['Annual_Benefits_Base_Rate','Employee_Code','Employee_GL_Code','Employee_Name','Department_Desc','Workers_Comp_Desc',
                    'Street','1099_Electronic_Only_Election','ACA_Electronic_Only_Election','Block_SUI','Fed_Addl_%','Fed_Deductions_$',
                    'Fed_Filing_Status_Description','W2_Electronic_Only_Election','Subdepartments/Markets_Desc','District_Code','Birth_Date_(MM/DD/YYYY)',
                    'Length_of_Service_Since_Hire','Length_of_Service_Since_Seniority','Zipcode','Initial_ACA_Status'],inplace=True)
print(df.info())


# 1.
print(df['Employee_Status'].value_counts())
#Dropping employee who ever never hired
df=df.loc[~(df['Employee_Status']=='Not Hired'),:]
print(df['Employee_Status'].value_counts())

#2.
print(df['City'].isna().value_counts())
print(df.loc[df['City'].isnull(),'City'])
#Only 1 record didnt had the city and zipcode. Dropping that row
df = df.loc[~(df['City'].isna()),:]

#3.
print(df['DOL_Status'].value_counts())
#Dropping Part timers as the data set is too small
df = df.loc[~(df['DOL_Status'].isna()),:]
#Blanks are all Full timers - Confirmed with Data expert
#dropping the column as all data set are now full timers
#######          Need to build Full and Part timers models to see if there is any coorelation
df.drop(columns='DOL_Status',inplace=True)

#4.
print(df['EEO1_Disabled_Status'].value_counts())
print(df['EEO1_Ethnicity'].value_counts())
#Keeping Disability and Ethnicity factors out data analysis scope
df.drop(columns=['EEO1_Disabled_Status','EEO1_Ethnicity'],inplace=True)

#5.
print(df['ESS_Gender'].value_counts())
#Dropping the columns since most of the data is "Unspecified or Dont wish to answer"
df.drop(columns='ESS_Gender',inplace=True)

#6.
print('Position')
print(df['Position'].value_counts())
#Dropping Retail Store Manager as the analysis scope is for below Retail Store Manager
df = df.loc[~(df['Position']=='Retail Store Manager'),:]
#Dropping Position column as all employees are now below Retail Store Manager
df.drop(columns='Position',inplace=True)

#7.
#Reviewing Each column data
print('Department')
print(df['Department'].value_counts())
df_term=df.loc[df['Employee_Status']=='Terminated',:]
print(df_term['Department'].value_counts())
#Removing Department as 1/3 of the data doesnt have Department name and same ratio of termniation
df.drop(columns='Department',inplace=True)

print('District_Desc')
print(df['District_Desc'].value_counts())
# Same issue as Department ... Employees not assinged to a district
df.drop(columns='District_Desc',inplace=True)

print(df['Labor_Allocation_Details'].value_counts())
# Same issue as Department ... Employees not assinged to a district
df.drop(columns='Labor_Allocation_Details',inplace=True)

print(df['ACA_Status'].value_counts())
# 1/3 of data is Not Measured.. 
df.drop(columns='ACA_Status',inplace=True)

print(df['Gender'].value_counts())
df.loc[df['Gender']=='I do not wish to self-identify','Gender']= 'Unspecified'
print(df['Gender'].value_counts())

# print(df['Time_in_Position'].value_counts()) - Need a conversion tool
print(df['Vets_4212_Emp_Category'].value_counts())

#90% of the data is Not Specified. Dropping the column
df.drop(columns='Vets_4212_Emp_Category',inplace=True)

#Adding colum the defines if Work and Live is in same city
df['Work_Live_Same_City']=df['Lives-in_State']==df['Works-in_State']


#Dropping Lives in State and Work in space column as i have identifed Work n Live in State r same or not. Also we have another State col
df.drop(columns=['Lives-in_State','Works-in_State'],inplace=True)

#Dropping Annual salary because Hrly rate is taken
df.drop(columns='Annual_Salary',inplace=True)

#Dropping data where Location is BLANK
df = df.loc[~(df['Last Location']==0),:]

#Changing "On Leave" Employment status to Active
df.loc[df['Employee_Status']=='On Leave','Employee_Status']= 'Active'


#Dropping Termination and Hire date becase we have Length_of_Service column
# df.drop(columns=['Termination_Date','Hire_Date'],inplace=True)

#----------------------------------------- Data Cleaning [END]-------------------------


#------------------------------------------EDA PART 1 [START]-------------------
#EDA Part 1 (Before Tranformation)

#---------- Finding the right data for the model [Removing some data that can be used in different ML model]

#Drop data where sales data is not available (We can use this data for other analysis)
df = df.loc[~(df['Month1'].isna()),:]

#There are employees who dont have 6 months of data.. 
#Limiting sales data to 3 months
df.drop(columns=['Month4','Month5','Month6'],inplace=True)

#Dropping employee who were hired within last 6 months
df=df.loc[~(df['Hire_Date']>'02/28/2020'),:]

#Dropping Termination and Hire date becase we have Length_of_Service column
df.drop(columns=['Termination_Date','Hire_Date'],inplace=True)


# df_not_active=df.loc[(df['Employee_Status']=='Terminated'),:]
# print(df_not_active.shape)
# print(df_not_active['Length_of_Service'].describe())


#'Finding balanced data to run the model'
print(df['Length_of_Service'].count())
df_90=df['Length_of_Service']>=90
print('Greater than 90: ',df.loc[df_90,'Length_of_Service'].count())
df['More_than_180']= df['Length_of_Service'].apply(lambda x: 0 if x <180 else 1)


# sns.boxplot(x='Employee_Status', y='Rate_1', data=df)
# # plt.xlabel('Cancellation Type',fontsize=10)
# # plt.ylabel('Lead Time',fontsize=10)
# # plt.title('Cancellation by Lead time',c='b',fontsize=16)
# plt.show()

# sns.boxplot(x='Employee_Status', y='Length_of_Service', data=df)
# plt.show()

# sns.boxplot(x='Employee_Status', y='Length_of_Service', data=df)
# plt.show()

# sns.scatterplot(x='Length_of_Service', y='Month1', data=df, ci=None, hue='Employee_Status')
# plt.show()


# sns.pairplot(df,
#              x_vars = ['Month1', 'Age'],
#              y_vars = ['Length_of_Service'],
#              hue = 'Employee_Status')
# plt.show()


# -------------------------



#Transform Categorical Values:


encoder = ce.OrdinalEncoder(cols=['score'],return_df=True,mapping=[{'col': 'Employee_Status','mapping': {'Terminated': 0,'Active' : 1}},
                                                                {'col': 'Work_Live_Same_City','mapping': {False: 0,True : 1}}])
newDF = encoder.fit_transform(df)

cols = ['Gender', 'City', 'State','Block_Fed_Tax?','Fed_Filing_Status','Fed_Multiple_Jobs?','Tobacco_User','Last Location']
newDF[cols] = newDF[cols].apply(LabelEncoder().fit_transform)

print(newDF.head())
print(newDF.info())

df=newDF

#--------------------

# EDA  (PArt 2):

#1



# # #Heatmap
# plt.figure(figsize = (8, 5))
# corr = df[['Rate_1','Employee_Status','Age','Length_of_Service','Fed_Dependents_$','Month1','Month2','Month3','Loc_Month1',
#         'Loc_Month2','Loc_Month3']].corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# sns.heatmap(corr, mask = mask, cmap = 'Blues', annot = True)
# plt.show()

# # #Histogram for Distribution
df.loc[:,:].hist(bins=25,
                 figsize=(8,5),
                 xlabelsize='10',
                 ylabelsize='10',xrot=-15)
plt.show()



# # #Scatter Plot
sns.scatterplot(x='Month1', y='Loc_Month1', data=df, ci=None,hue='Employee_Status')
plt.show()

fig, axes = plt.subplots(nrows = 1,ncols = 3,figsize = (8,2))
sns.scatterplot(x='Month1', y='Loc_Month1', data=df, ci=None, ax = axes[0], hue='Employee_Status')
sns.scatterplot(x='Month2', y='Loc_Month2', data=df, ci=None, ax = axes[1], hue='Employee_Status')
sns.scatterplot(x='Month3', y='Loc_Month3', data=df, ci=None, ax = axes[2], hue='Employee_Status')
fig.tight_layout()
plt.show()


sns.pairplot(df,
             x_vars = ['City', 'Age','Month1','Loc_Month1'],
             y_vars = ['Length_of_Service'],
             hue = 'Employee_Status')
plt.show()

#Droppping Employee Status data for testing
df.drop(columns='Employee_Status',inplace=True)
df.drop(columns='Length_of_Service',inplace=True)


#--------------- ML

X=df.drop(columns='More_than_180')
y=df['More_than_180']


#instantiate
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3,stratify=y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)




#-----------------------------------------------------------------


# KNN:

from sklearn.neighbors import KNeighborsClassifier

# Hyper Tuning Parameters:

# max_depth_length = list(range(2,9))
# accuracy = []
# for depth in max_depth_length:
#     clf=KNeighborsClassifier(n_neighbors=depth)
#     clf.fit(X_train,y_train)
#     score = clf.score(X_test,y_test)
#     accuracy.append(score)

# plt.plot(max_depth_length,accuracy)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
predict = knn.predict(X_train)
score=knn.score(X_test,y_test)
print('KNN Score: ',score)


# def RandomForest():
    
# Random Forest:
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# Hyper Tuning Parameters:

# # Randomized & Grid Search
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=3,stratify=y)
# clf = RandomForestClassifier(random_state=3)

# param_grid={'n_estimators': [25,50,100,125,200],
#         'min_samples_split': [5,6,7,8],
#             'max_depth': [2,3,4,5]}

# cv_rs=RandomizedSearchCV(estimator=clf,param_distributions=param_grid,cv=3,n_iter = 10,random_state=3)
# cv_rs.fit(X_train,y_train)
# print('Random Search : ',cv_rs.best_params_)     

# cv_clf=GridSearchCV(estimator=clf,param_grid=param_grid,cv=3)
# cv_clf.fit(X_train,y_train)
# print('Best Parameter')
# print(cv_clf.best_params_)


clf = RandomForestClassifier(n_estimators=125,max_depth=5,min_samples_split=5,bootstrap=True,oob_score=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Random Forest Classifier Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Random Forest Classifier Test: ',score_test)


# #Decision Tree:

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


# Hyper Tuning Parameters:

# max_depth_length = list(range(1,6))
# accuracy = []
# for depth in max_depth_length:
#     clf=DecisionTreeClassifier(max_depth=depth,random_state=3)
#     clf.fit(X_train,y_train)
#     score = clf.score(X_test,y_test)
#     accuracy.append(score)

# plt.plot(max_depth_length,accuracy)
# plt.show()

clf = DecisionTreeClassifier(max_depth = 2, random_state = 3)
clf.fit(X_train,y_train)
clf.predict(X_test)
print('Decision Tree Train: ',clf.score(X_train,y_train))
print('Decision Tree Test: ',clf.score(X_test,y_test))


# #Logistic Regression L1

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1,penalty='l1', solver='liblinear',multi_class='ovr')
log_reg.fit(X_train,y_train)
print('Logistic Regression L1 Training Accuracy: ',log_reg.score(X_train,y_train))
print('Logistic Regression L1 Testing Accuracy: ',log_reg.score(X_test,y_test))

# #Logistic Regression L2

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1,penalty='l2', solver='liblinear',multi_class='ovr')
log_reg.fit(X_train,y_train)
print('Logistic Regression L2 Training Accuracy: %.2f' % log_reg.score(X_train,y_train))
print('Logistic Regression L2 Testing Accuracy: ',log_reg.score(X_test,y_test))


# #LinearRegression

from sklearn.linear_model import LinearRegression
clf = LinearRegression(fit_intercept=True)
clf.fit(X_train,y_train)
clf.predict(X_test)
score_train=clf.score(X_train,y_train)
print('Linear Train: ',score_train)    
score_test=clf.score(X_test,y_test)
print('Linear Test: ',score_test)


