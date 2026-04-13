import pandas as pd
import numpy as np
import ipaddress

def load_and_process_data():
    df=pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
        
    print("---------------- exploring the data... ----------------\n")
    
    df=pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

    print(df.head())
    print("number of rows is :",len(df.index))

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
    print(df["LastLoginIP"].head())

    print("----------------AccountStatus, Churn------------------")
    print(df.columns.unique())
    print(df["AccountStatus"].head(30))
    print("unique values:")
    print(len(df["AccountStatus"].unique()))
    print("possible values of AccountStatus:", sorted(df["AccountStatus"].unique()))
    print("frequencies of AccountStatus:")
    print(df["AccountStatus"].value_counts())


    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # Panier moyen
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']

    # Anciennete vs activite recente
    df['TenureRatio'] = df['Recency'] / df['CustomerTenureDays']


    df.drop(columns=["MonetaryTotal", "Recency", "Frequency", "CustomerTenureDays"], inplace=True)


    # print("----------------GeoIP Numeric Feature Engineering------------------")

    # import ipaddress

    # def ip_to_int(ip):
    #     try:
    #         return int(ipaddress.IPv4Address(ip))
    #     except:
    #         return None


    # def cidr_to_range(cidr):
    #     net = ipaddress.ip_network(cidr, strict=False)
    #     return int(net.network_address), int(net.broadcast_address)


    # # Load datasets
    # blocks = pd.read_csv("data/external/GeoLite2-Country-Blocks-IPv4.csv")
    # locations = pd.read_csv("data/external/GeoLite2-Country-Locations-en.csv")

    # # Keep only needed columns
    # blocks = blocks[["network", "geoname_id"]]
    # locations = locations[["geoname_id", "country_iso_code"]]

    # # Merge
    # geo_df = blocks.merge(locations, on="geoname_id", how="left")

    # # Convert CIDR → range
    # geo_df["ip_start"] = geo_df["network"].apply(lambda x: cidr_to_range(x)[0])
    # geo_df["ip_end"] = geo_df["network"].apply(lambda x: cidr_to_range(x)[1])


    # def get_country(ip):
    #     ip_int = ip_to_int(ip)
    #     if ip_int is None:
    #         return "UNK"

    #     match = geo_df[(geo_df["ip_start"] <= ip_int) & (geo_df["ip_end"] >= ip_int)]

    #     if match.empty:
    #         return "UNK"

    #     return match.iloc[0]["country_iso_code"]


    # # Map IP → country code
    # df["Country"] = df["LastLoginIP"].apply(get_country)

    # # Convert directly to numeric features
    # df["Country"] = df["Country"].astype("category")
    # df["Country_encoded"] = df["Country"].cat.codes

    # country_freq = df["Country"].value_counts(normalize=True)
    # df["Country_freq"] = df["Country"].map(country_freq)

    # # FINAL CLEANUP → remove redundancy
    # df.drop(columns=["LastLoginIP", "Country"], inplace=True)

    # print("Final numeric features added successfully")
    # print(df[["Country_encoded", "Country_freq"]].head())

    print("----------------Class imbalance check------------------")

    print("Churn distribution:")
    print(df["Churn"].value_counts(normalize=True))

    print("\nAccountStatus distribution:")
    print(df["AccountStatus"].value_counts(normalize=True))
    #we will drop the account status column
    df.drop(columns=["AccountStatus"], inplace=True)
    df.drop(columns=["CustomerID"], inplace=True)
    df.drop(columns=["RegistrationDate"], inplace=True)

    return df


if __name__ == "__main__":
    # When running as a script, save the processed dataset
    df_processed = load_and_process_data()
    df_processed.to_csv("data/processed/retail_customers_processed.csv", index=False)
    print("Processed dataset saved to data/processed/retail_customers_processed.csv")
    # print(df_processed.head())
    # print("number of rows is :",len(df_processed.index))
    # print(df_processed.columns.unique())
    #print(df_processed["Churn"].value_counts())