import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")

print(pd.__version__)  # Q1

print(f'Number of records in the dataset: {df.shape[0]}')    # Q2

print(f'Number of laptop\'s brands in the dataset: {df['Brand'].nunique()} \n {df.Brand.value_counts()}')    # Q3

print(f'Columns with NAs: \n {df.isna().sum()}')    # Q4

print(f'Maximum price for a Dell notebook: {df.loc[df.Brand.eq('Dell'), 'Final Price'].max()}')    # Q5

first_median = df['Screen'].median()
mode = df['Screen'].mode()
updated_median = df['Screen'].fillna(mode).median()
print(first_median == updated_median)    # Q6
