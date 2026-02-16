import pandas as pd 
df = pd.read_csv('../retail_customers_COMPLETE_CATEGORICAL.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
print(sorted(df["Age"].unique())) 
print((df["SupportTicketsCount"].unique()))
