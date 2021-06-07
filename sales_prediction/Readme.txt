Sales Prediction for an online store

1. Introduction:

1.1 Problem: 
	Predicting sale is one of the important goal for any investor / business entity. 
	If we can accurately predict sales, it gives the owner / business to predict inventory and / or make future investments

1.2 Objective:
	The objective is to forecast sales

1.3 Data Used:
	Data has been taken from 

1.4 Tools used:
	Python, Multiple regression techniques - Linear, KNN and Random Forest , Feature Importance 

2 Data Wrangling:

2.1 Find BLANKS
	During the initial data analysis we found missing data for Weight and Outlet Size

2.2 Act on BLANK
	There are no corelation to fill Item weight and was dropped from the data set
	There are corelation found Based on Outlet type and and SuperMarket Type1 was considered as Small size

2.3 Formatting
	Multiple data points had naming issues
	- LF was replaced with Low Fat
	- reg was replaced with Regular

2.4 Derived Features (Data transformation)
	- Year columns was grouped in a 10 year span and added to a new column "no_of_years"
	- Outlet Establishment Year column was then dropped
	- Ordinal Columns were transformed
		- 'Item_Fat_Content'
		- 'Outlet_Size'
		- 'Outlet_Location_Type'
		- 'Outlet_Type'
		- 'no_of_years'
	- Nominal Columns were transformed
		- 'Item_Visibility'
		- 'Item_MRP'
		- 'Item_Outlet_Sales'


3. EDA (Exploratory Data Analysis)

3.1 Average Sales by Outlet Type 
3.2 Top and Bottom Store in Sales
3.3 Sales for all Stores
3.4 Volume of sale by Price
3.5 Profit based on Outlet Type

4. Machine Learning

4.1 Train Test data was used on 3 Regressor models
	- Linear
	- KNN
	- Random 

4.2 Hyper parameters Tuning
	- Feature Importance was explored to identify which Data columns matter the most
	- Grid Search was used to find the best (RandomForestRegressor)
