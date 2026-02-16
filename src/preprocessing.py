import pandas as pd 
df = pd.read_csv('../retail_customers_COMPLETE_CATEGORICAL.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
print(sorted(df["Age"].unique())) 
print((df["SupportTicketsCount"].unique()))
df = df.drop(columns=["NewsletterSubscribed"])
print("-------------------")
try: 
    print((df["NewsletterSubscribed"].unique()))
except : print("Column 'NewsletterSubscribed' has been dropped and is no longer available.")