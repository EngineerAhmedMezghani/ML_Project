import pandas as pd

df = pd.read_csv("data/processed/retail_customers_processed.csv")
print(df.head())

print(df.columns.unique())
for i in df.columns.unique():
    print(i)
    print(df[i].dtype)
    print("---")
