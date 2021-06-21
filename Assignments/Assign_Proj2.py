import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

df=pd.read_excel(r"C:\Users\dangs\Downloads\Arch_Employee_Data.xlsx")
print(df.head())
print(df.info())

#Deleting data that has lot of missing data
df=df.drop(columns=['Work_Location','Position_Code','Employment_Predictor_Score','Act._Marital_Status',
                        'Exempt_Status','Length_of_Service_Since_Rehire','Position_Family','Business_Title','#State_Exemptions/Allowances',
                        'Block_State_Tax?','Estimated_Deductions','State_Filing_Status','State_Filing_Status_Desc','W2_Info_12DD_(2020)',
                        'RehirefromT-Mobile','ESS_EEO1_Ethnicity/Race','Supervisor_Approval','Supervisor_Approval_Code','Budgeted',
                        'Supervisor_Primary','Supervisor_Primary_Code','Supervisor_Secondary','Supervisor_Secondary_Code','Supervisor_Tertiary',
                        'Supervisor_Tertiary_Code','TerminationCategory','TrainingLocation','Current_Key_Employee','ESS_Language_Preference',
                        'EEO1_Category','Termination_Reason','Termination_Type','Business_Title.1'])

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

#Dropping Termination and Hire date becase we have Length_of_Service column
df.drop(columns=['Termination_Date','Hire_Date'],inplace=True)

#Dropping Lives in State and Work in space column as i have identifed Work n Live in State r same or not. Also we have another State col
df.drop(columns=['Lives-in_State','Works-in_State'],inplace=True)


print(df.info())

# 
#90 days more or less
#Model Selection :
    # How long is the tenure
    #Likelyhood of them leaving
# Feature importance
