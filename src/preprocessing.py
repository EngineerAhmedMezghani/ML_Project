import pandas as pd
import numpy as np

def load_and_process_data():
    df=pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
        
    print("---------------- exploring the data... ----------------\n")
    
    df=pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

    print(df.head())
    print(len(df.columns))

    print("----------------fixing age column...------------------\n")
    print(df["Age"].isnull().value_counts())

    skewness = df['Age'].skew()
    print("Skewness of Age:", skewness)
    #skew to see the symetry of the age values 

    #it's almost symetric => applying mean 
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    print("creating a new dataframe with filled age column...")
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    print(df["Age"].isnull().value_counts())
    print("column age is now filled with mean values")




    print("----------------SupportTickets, Satisfaction------------------\n")


    print("min SupportTicketsCount:", min(df["SupportTicketsCount"]))
    print("max SupportTicketsCount:", max(df["SupportTicketsCount"]))
    print("value counts SupportTicketsCount:\n", df["SupportTicketsCount"].value_counts())
    skewness = df['SupportTicketsCount'].skew()
    print("Skewness of SupportTicketsCount", skewness)

    print("min SatisfactionScore:", min(df["SatisfactionScore"]))
    print("max SatisfactionScore:", max(df["SatisfactionScore"]))
    print("value counts SatisfactionScore:\n", df["SatisfactionScore"].value_counts())
    skewness = df['SatisfactionScore'].skew()
    print("Skewness of SatisfactionScore", skewness)


    # SupportTicketsCount: median
    print("\nfilling with the central value of the range [0,7]\n")
    median_support = df.loc[df["SupportTicketsCount"].between(0,7), "SupportTicketsCount"].median()
    df.loc[~df["SupportTicketsCount"].between(0,7), "SupportTicketsCount"] = median_support
    print("value counts SupportTicketsCount:\n", df["SupportTicketsCount"].value_counts())
    skewness = df['SupportTicketsCount'].skew()
    print("Skewness of SupportTicketsCount after median", skewness)

    # SatisfactionScore: mode
    print("\nfilling with the most frequent value in the range [0,5]\n")
    mode_satisfaction = df.loc[df["SatisfactionScore"].between(0,5), "SatisfactionScore"].mode()[0]
    df.loc[~df["SatisfactionScore"].between(0,5), "SatisfactionScore"] = mode_satisfaction
    print("value counts SatisfactionScore:\n", df["SatisfactionScore"].value_counts())
    skewness = df['SatisfactionScore'].skew()
    print("Skewness of SatisfactionScore after mode", skewness)

    print("----------------registration date------------------")
    print(df["RegistrationDate"].head())
    # Convert RegistrationDate to datetime with day first
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], 
        format="%d/%m/%y",  # day/month/two-digit year
        errors="coerce"
    )
    df["RegistrationDate_day"] = df["RegistrationDate"].dt.day
    df["RegistrationDate_month"] = df["RegistrationDate"].dt.month
    df["RegistrationDate_year"] = df["RegistrationDate"].dt.year
    print(df[["RegistrationDate", "RegistrationDate_day", "RegistrationDate_month", "RegistrationDate_year"]])



    print("----------------NewsletterSubscribed------------------")
    df.drop(columns=["NewsletterSubscribed"], inplace=True)
    print(df.describe())
    print(len(df.columns))


    print("----------------LastLoginIP------------------")
    print("unique values:")
    print(len(df["LastLoginIP"].unique()))

    print("----------------AccountStatus, Churn------------------")
    return df



if __name__ == "__main__":
    # When running as a script, save the processed dataset
    df_processed = load_and_process_data()
    df_processed.to_csv("data/processed/retail_customers_processed.csv", index=False)
    print("Processed dataset saved to data/processed/retail_customers_processed.csv")