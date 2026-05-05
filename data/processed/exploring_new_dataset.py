import pandas as pd

df = pd.read_csv("data/processed/retail_customers_processed.csv")
print(df.head())


for col in df:
    print(f"{col}: {sum(df[col].isnull())}")

print(df.shape)