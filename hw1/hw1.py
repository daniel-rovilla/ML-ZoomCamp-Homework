import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")

print(pd.__version__)   # Q1

print(f'Number of records in the dataset: {df.shape[0]}')   # Q2

print(f'Number of laptop\'s brands in the dataset: {df['Brand'].nunique()} \n {df.Brand.value_counts()}')   # Q3

print(f'Columns with NAs: \n {df.isna().sum()}')  # Q4
