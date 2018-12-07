import numpy as np
import pandas as pd

# Examples from https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python#question1
# https://pandas.pydata.org/pandas-docs/stable/10min.html
# https://medium.com/@harrypotter0/an-introduction-to-data-analysis-with-pandas-27ecbce2853

# DataFrames Definition
data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])

print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))

my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
print(pd.DataFrame(my_2darray))

# Remember --> when it's a dictionary, the columns are sorted (mainly when the keys are numbers)
my_dict = {'1': ['1', '3'], '99': ['1', '2'], '3': ['2', '4']}
print(pd.DataFrame(my_dict))

my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['A'])
print(my_df.shape)
print(my_df.columns.values)
print(my_df.index.values)



my_series = pd.Series({"Belgium":"Brussels", "India":"New Delhi", "United Kingdom":"London", "United States":"Washington"})
print(pd.DataFrame(my_series))

# pd.DataFrame.index: height of DataFrame
# pd.DataFrame.columns: width of DataFrame

# Selecting elements in a DataFrame
# .loc[] - works on labels of your index. ex: loc[2], index labeled 2
#
# .iloc[] - works on the positions of your index. iloc[2], index at 2
#
# .ix[] - when the index is integer-based, you pass a label to .ix[] x[2] then
# means that you’re looking in your DataFrame for values that have an index
# labeled 2. This is just like .loc[]! However, if your index is not solely
# integer-based, ix will work with positions, just like .iloc[].
#
# Dictionary, keys become columns' labels.
df = pd.DataFrame({'A':[1,4,7],'B':[2,5,8],'C':[3,6,9]})
print(df.iloc[0][0]) # important notation

# Using `loc[]`
print(df.loc[0]['A']) # important notation

# Using `at[]`
print(df.at[0,'A'])

# Using `iat[]`
print(df.iat[0,0])

# Use `iloc[]` to select row `0`
print(df.iloc[0])

# Use `loc[]` to select column `'A'`
print(df.loc[:,'A'])

# df[column][row]
# Rows can receive a number (position/index) instead of the exact label



## Adding rows and columns
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])

# There's no index labeled `2`, so you will change the index at position `2`
df.ix[2] = [60, 50, 40]
print(df)

# This will make an index labeled `2` and add the new values
# Loc -> it the 2 doesn't exist, the DataFrame will create a row with label 2.

# Adding a row
df.loc[2] = [11, 12, 13]
# Adding a column
print(df)

# Append column with Pandas Series
df.loc[:, 9.9] = pd.Series([99,88,77,33], index=df.index)
print(df)


## Resetting the Index of Your DataFrame
df.reset_index(level=0,drop=True)

## Deleting indices, rows or columns

# Removing duplicates rows
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [40, 50, 60], [23, 35, 37]]),
                  index= [2.5, 12.6, 4.8, 4.8, 2.5],
                  columns=[48, 49, 50])
#
# df.reset_index.drop_duplicates(subset='index', keep='last').set_index('index')
#
# #  Deleting column
# df.drop('48',axis=1,inplace=True)

# Drop the column with label 48
df.drop(48, axis=1, inplace=True)

# Drop the column at position 1
df.drop(df.columns[[1]], axis=1)

# Dropping a row (index)
# df.drop(df.index[_])

# Generic drop (column or row)
# df.drop(Label)

# del df[48]


# Formatting the data
# Replacing Strings. It's like mapping values
df = pd.DataFrame(data=[['OK', 'Perfect', 'Acceptable'],['Awful','Awful','Perfect']],columns=['St1','St2','St3'])
print(df)
print(df.replace(['Awful', 'OK', 'Acceptable', 'Perfect'], [3, 1, 2,4]))

# Applying A Function to Your Pandas DataFrame’s Columns or Rows
print(df['St1'].apply(lambda s: s + 'AAA'))

# doubler = lambda x: x*2
# df['A'].apply(doubler)
# df.loc[0].apply(doubler)
#Whole DataFrame
# doubled_df = df.applymap(doubler)
# def doubler(x):
#     if x % 2 == 0:
#         return x
#     else:
#         return x * 2
#
# # Use `applymap()` to apply `doubler()` to your DataFrame
# doubled_df = df.applymap(doubler)
#
# # Check the DataFrame
# print(doubled_df)

# Empty DataFrame
df = pd.DataFrame(np.nan, index=[0,1,2,3], columns=['A'])
print(df)

# Importing Data
# pd.read_csv('yourFile', parse_dates=True)
pd.read_csv('./data/iris.csv')

#Reshaping DataFrame
## Pivotting --> create a new derived table out of your original one
products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
                        'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
                        'price':[11.42, 23.50, 19.99, 15.95, 55.75, 111.55],
                        'testscore': [4, 3, 5, 7, 5, 8]})

# Use `pivot()` to pivot the DataFrame
pivot_products = products.pivot(index='category', columns='store', values='price')

# Check out the result
print(pivot_products)

# Melting is considered useful in cases where you have a data that has one or
# more columns that are identifier variables, while all other columns are
# considered measured variables.
# The `people` DataFrame
people = pd.DataFrame({'FirstName' : ['John', 'Jane'],
                       'LastName' : ['Doe', 'Austen'],
                       'BloodType' : ['A-', 'B+'],
                       'Weight' : [90, 64]})

# Use `melt()` on the `people` DataFrame
print(pd.melt(people, id_vars=['FirstName', 'LastName'], var_name='measurements'))

# Iterating
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])

for index, row in df.iterrows() :
    print(row['A'], row['B'])


# Exporting file
df.to_csv('./data/teste.csv',sep='\t')

writer = pd.ExcelWriter('./data/myDataFrame.xlsx')
df.to_excel(writer, 'DataFrame')
writer.save()

# https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/
# Selection based on condtions

# Create a dataframe
raw_data = {'first_name': ['Jason', 'Molly', np.nan, np.nan, np.nan],
        'nationality': ['USA', 'USA', 'France', 'UK', 'UK'],
        'age': [42, 52, 36, 24, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'nationality', 'age'])
# Create variable with TRUE if nationality is USA
american = df['nationality'] == "USA"

# Create variable with TRUE if age is greater than 50
elderly = df['age'] > 50

# Select all cases where nationality is USA and age is greater than 50
#result = df.loc[df[american & elderly]] # == df[df[american & elderly]]

#  Piece of DataFrame
df.tail(2) # the last 2;
df.head(2) # the 2 first

# Some basic information about a DataFrame
df.info()

# Missing data
df.dropna(how='any') #drop eows with missing data
df.fillna(value=5) # filling the missing data with value 5
pd.isna(df) # checking missing values

# Basic operations
df.mean()

# Histogramming
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

#Time Series
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()


# Group By
# df.groupby('some_label').sum()

#Plotting
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
