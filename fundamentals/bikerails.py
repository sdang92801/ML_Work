import pandas as pd
df = pd.read_excel(r'C:\Users\Employee\Downloads\RailsToTrails.xlsx')
print(df)
# Select just the data after July 31, 2020
after_filter = df['Week of']> '2020-07-31'
print(df.loc[after_filter,:])

# Select just data where the 'Change 2019-2020' column is greater than 100% (greater than 1)
greater_filter = df['Change 2019-2020'] > 1
print(df.loc[greater_filter,:])

# Both filter in place
print(df.loc[after_filter & greater_filter,:])
