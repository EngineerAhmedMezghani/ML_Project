import pandas as pd

df = pd.read_csv("data/processed/retail_customers_processed.csv")
print(df.head())



print(df["Gender_encoded"].unique())
print(df["Gender_encoded"].value_counts())

print(df["Region_encoded"].unique())
print("Region_encoded value counts:",df["Region_encoded"].value_counts())

print(df["FavoriteSeason_encoded"].unique())
print("FavoriteSeason_encoded value counts:",df["FavoriteSeason_encoded"].value_counts())

print(df["PreferredTimeOfDay_encoded"].unique())
print("PreferredTimeOfDay_encoded value counts:",df["PreferredTimeOfDay_encoded"].value_counts())

print(df["WeekendPreference_encoded"].unique())
print("WeekendPreference_encoded value counts:",df["WeekendPreference_encoded"].value_counts())

print(df["CustomerType_encoded"].unique())
print("CustomerType_encoded value counts:",df["CustomerType_encoded"].value_counts())

label_cols = [
    "LoyaltyLevel",
    "AgeCategory",
    "SpendingCategory",
    "ChurnRiskCategory",
    "BasketSizeCategory",
    "ProductDiversity",
    "RFMSegment"
]
for col in label_cols:
    print(df[col].unique())
    print(f"{col} value counts:",df[col].value_counts())
    print("---")

