import pandas as pd

df = pd.read_csv("data/processed/retail_customers_processed.csv")
print(df.head())

print(df.columns.unique())
