import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")

# What's the version of Pandas that you installed?
print(pd.__version__)  # Q1

# How many records are in the dataset?
print(f'Number of records in the dataset: {df.shape[0]}')  # Q2

# How many laptop brands are presented in the dataset?
print(f'Number of laptop\'s brands in the dataset: {df['Brand'].nunique()} \n {df.Brand.value_counts()}')  # Q3

# How many columns in the dataset have missing values?
print(f'Columns with NAs: \n {df.isna().sum()}')  # Q4

# What's the maximum final price of Dell notebooks in the dataset?
print(f'Maximum price for a Dell notebook: {df.loc[df.Brand.eq('Dell'), 'Final Price'].max()}')  # Q5

# Find the median value of Screen column in the dataset.
# Next, calculate the most frequent value of the same Screen column.
# Use fillna method to fill the missing values in Screen column with the most frequent value from the previous step.
# Now, calculate the median value of Screen once again.
# Has it changed?
first_median = df['Screen'].median()
mode = df['Screen'].mode()
updated_median = df['Screen'].fillna(mode).median()
print(first_median == updated_median)  # Q6

# Select all the "Innjoo" laptops from the dataset.
# Select only columns RAM, Storage, Screen.
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# Compute the inverse of XTX.
# Create an array y with values [1100, 1300, 800, 900, 1000, 1100].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the sum of all the elements of the result?

X = df.loc[df['Brand'] == 'Innjoo', ['RAM', 'Storage', 'Screen']].to_numpy()    # X = df.loc[df['Brand'].eq('Innjoo'), ['RAM', 'Storage', 'Screen']].to_numpy() / Different approach, same result
X_transpose = X.T  # Transpose of X
XTX = X_transpose @ X  # Matrix-matrix multiplication between X.T and X
XTX_inv = np.linalg.inv(XTX)  # Compute the inverse of XTX.
y = np.array([1100, 1300, 800, 900, 1000, 1100])    # Create the array y
w = XTX_inv @ X_transpose @ y  # Compute w
print(np.sum(w))    # Q7
